from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import json
import numpy as np
import os

# Import de tes propres fonctions
from qdrant_tools import rechercher_reactions_similaires
from analyse import modeliser_recompense_semantique, optimiser_routage_topsis, deriver_poids_ahp

from qdrant_tools import rechercher_reactions_similaires, indexer_corpus_generique
from qdrant_client.models import Distance


# --- Variables Globales ---
ml_models = {}
qdrant_db = {}
app_data = {}


def seuil_semantique_defaut() -> float:
    valeur = os.getenv("SEMANTIC_SCORE_THRESHOLD", "0.5")
    try:
        return float(valeur)
    except ValueError:
        return 0.5


def infos_analyse(resultats: List[Dict[str, Any]], seuil: float) -> Dict[str, Any]:
    volumes = {}
    scores = []
    for r in resultats:
        modele = r.get("refers_to_model")
        if modele:
            volumes[modele] = volumes.get(modele, 0) + 1
        if r.get("score") is not None:
            scores.append(r["score"])

    max_support = max(volumes.values()) if volumes else 0
    support_faible = max_support < 3
    similarite_moyenne = round(sum(scores) / len(scores), 4) if scores else 0.0
    avertissement = None

    if support_faible:
        avertissement = (
            "Peu de données similaires ont été trouvées. Le résultat est indicatif : "
            "cela peut venir du seuil de similarité ou simplement d'une base de données trop limitée."
        )

    return {
        "score_threshold": seuil,
        "nb_reactions_similaires": len(resultats),
        "nb_modeles_couverts": len(volumes),
        "max_volume_support": max_support,
        "similarite_moyenne": similarite_moyenne,
        "support_faible": support_faible,
        "avertissement": avertissement,
    }


def calculer_modeles_exclus(
    resultats_semantiques: Dict[str, Dict[str, Any]],
    metriques_physiques: Dict[str, Dict[str, Any]],
    noms_criteres: List[str]
) -> Dict[str, Any]:
    exclus = []
    raisons = {"metriques_physiques_manquantes": 0, "criteres_incomplets": 0}

    for modele, donnees_semantiques in resultats_semantiques.items():
        donnees_physiques = metriques_physiques.get(modele)
        if donnees_physiques is None:
            raisons["metriques_physiques_manquantes"] += 1
            exclus.append({"modele": modele, "raison": "metriques_physiques_manquantes"})
            continue

        donnees_fusionnees = {**donnees_semantiques, **donnees_physiques}
        criteres_manquants = [c for c in noms_criteres if donnees_fusionnees.get(c) is None]
        if criteres_manquants:
            raisons["criteres_incomplets"] += 1
            exclus.append({
                "modele": modele,
                "raison": "criteres_incomplets",
                "criteres_manquants": criteres_manquants
            })

    return {
        "total": len(exclus),
        "raisons": {cle: valeur for cle, valeur in raisons.items() if valeur > 0},
        "liste": exclus
    }


def detailler_classement_topsis(
    classement: List[Any],
    resultats_semantiques: Dict[str, Dict[str, Any]],
    metriques_physiques: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    detail = []
    for modele, score_topsis in classement:
        donnees_semantiques = resultats_semantiques.get(modele, {})
        donnees_physiques = metriques_physiques.get(modele, {})
        detail.append({
            "modele": modele,
            "score_topsis": score_topsis,
            "score_semantique": donnees_semantiques.get("score_semantique"),
            "volume_support": donnees_semantiques.get("volume_support"),
            "similarite_moyenne": donnees_semantiques.get("similarite_moyenne"),
            "niveau_confiance": donnees_semantiques.get("niveau_confiance"),
            "kwh_token": donnees_physiques.get("kwh/token"),
            "score_souverainete": donnees_physiques.get("score_souverainete"),
        })
    return detail


def index_corpus(client, dim_vecteur):
    colonnes_payload_cibles = [
        "question_content", 
        "conversation_pair_id", 
        "refers_to_model", 
        "model_pos",
        "liked", 
        "disliked", 
        "comment", 
        "useful", 
        "creative", 
        "clear_formatting", 
        "superficial", 
        "instructions_not_followed", 
        "incorrect"
    ]
    indexer_corpus_generique(
        client=client, # Instance pré-existante du client Qdrant
        vector_file_path="./base_vectorielle/base_vectorielle_reactions_question_content.parquet",
        metadata_file_path="./database/reactions.parquet",
        collection_name="index_reactions_question_content",
        vector_size=dim_vecteur,
        vector_column="embedding", # Paramètre par défaut, rendu explicite ici
        join_key="id",             # Paramètre par défaut, rendu explicite ici
        payload_columns=colonnes_payload_cibles,
        index_fields=["conversation_pair_id"],
        batch_size=1000,           # Correspond au seuil de bufferisation initial
        distance_metric=Distance.COSINE
    )


# --- Gestion de la durée de vie (Lifespan) ---
# C'est la méthode moderne de FastAPI pour charger les modèles lourds au démarrage
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Démarrage du serveur : Chargement du modèle SentenceTransformer...")
    ml_models["encoder"] = SentenceTransformer("BAAI/bge-m3")
    
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    print(f"⏳ Démarrage du serveur : Connexion à Qdrant sur {qdrant_url}...")
    qdrant_db["client"] = QdrantClient(url=qdrant_url)
    index_corpus(qdrant_db["client"], dim_vecteur=1024)
    
    print("⏳ Démarrage du serveur : Chargement des métriques physiques...")
    with open('metriques_physiques.json', 'r', encoding='utf-8') as fichier:
        app_data["metriques_physiques"] = json.load(fichier)
        
    print("✅ Serveur prêt à recevoir des requêtes !")
    yield
    
    # Nettoyage à l'arrêt du serveur
    print("🛑 Arrêt du serveur : Libération des ressources.")
    ml_models.clear()
    qdrant_db.clear()
    app_data.clear()

# --- Initialisation de l'API ---
app = FastAPI(
    title="API IA Culturelles - Routeur Sémantique",
    description="API permettant d'évaluer le meilleur modèle LLM pour un prompt donné.",
    lifespan=lifespan
)

# --- Schémas de données (Pydantic) ---
# Définit la structure exacte attendue en entrée (le JSON que le front-end enverra)
class PromptRequest(BaseModel):
    prompt: str
    limit: int = 1000
    score_threshold: float | None = None

class RoutageRequest(BaseModel):
    prompt: str
    # Matrice AHP 3x3 par défaut (Sémantique, Énergie, Souveraineté)
    matrice_ahp: List[List[float]]
    limit: int = 1000
    score_threshold: float | None = None

# --- Endpoints ---
@app.post("/api/evaluer_prompt")
async def evaluer_prompt(request: PromptRequest):
    """
    Reçoit un prompt, cherche les réactions similaires et calcule la récompense sémantique.
    """
    try:
        model = ml_models["encoder"]
        client = qdrant_db["client"]

        # 1. Encodage du prompt
        vecteur = model.encode(request.prompt, convert_to_tensor=False).tolist()
        seuil = request.score_threshold if request.score_threshold is not None else seuil_semantique_defaut()

        # 2. Recherche Vectorielle (Qdrant)
        resultats = rechercher_reactions_similaires(
            client=client,
            vecteur_requete=vecteur,
            collection_name="index_reactions_question_content",
            limit=request.limit,
            score_threshold=seuil
        )

        # 3. Groupement des question_content + score par modèle (avant agrégation)
        questions_par_modele: dict = {}
        for r in resultats:
            modele = r.get("refers_to_model")
            question = r.get("question_content")
            score = r.get("score")
            if modele and question:
                if modele not in questions_par_modele:
                    questions_par_modele[modele] = []
                already = [e["question"] for e in questions_par_modele[modele]]
                if question not in already and len(questions_par_modele[modele]) < 3:
                    questions_par_modele[modele].append({"question": question, "score": score})

        # 4. Analyse Sémantique
        if not resultats:
            return {
                "message": "Aucune similarité trouvée.",
                "prompt": request.prompt,
                "recompenses": {},
                "questions_par_modele": {},
                "infos_analyse": infos_analyse(resultats, seuil)
            }

        analyse = modeliser_recompense_semantique(resultats)

        # 5. Retour au front-end
        return {
            "message": "Analyse réussie",
            "prompt": request.prompt,
            "recompenses": analyse,
            "questions_par_modele": questions_par_modele,
            "infos_analyse": infos_analyse(resultats, seuil)
        }

    except Exception as e:
        # En cas d'erreur (Qdrant éteint, etc.), on renvoie une erreur 500 propre
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/meilleur_modele")
async def obtenir_meilleur_modele(request: RoutageRequest):
    """
    Détermine le meilleur modèle d'IA en fonction du prompt et des préférences utilisateur (AHP).
    Critères pris en compte : [Score Sémantique, kWh/token, Score Souveraineté]
    """
    try:
        model = ml_models["encoder"]
        client = qdrant_db["client"]
        metriques_physiques = app_data["metriques_physiques"]

        # 1. Encodage et Recherche Vectorielle
        vecteur = model.encode(request.prompt, convert_to_tensor=False).tolist()
        seuil = request.score_threshold if request.score_threshold is not None else seuil_semantique_defaut()
        resultats = rechercher_reactions_similaires(
            client=client,
            vecteur_requete=vecteur,
            collection_name="index_reactions_question_content", # ou _comment selon ton choix
            limit=request.limit,
            score_threshold=seuil
        )

        if not resultats:
            return {
                "message": "Aucune donnée sémantique trouvée pour ce prompt.",
                "prompt": request.prompt,
                "modele_recommande": None,
                "score_topsis": None,
                "classement_complet": [],
                "classement_detaille": [],
                "questions_par_modele": {},
                "poids_criteres": {},
                "modeles_exclus": {"total": 0, "raisons": {}, "liste": []},
                "infos_analyse": infos_analyse(resultats, seuil)
            }

        # 2. Groupement des question_content + score par modèle (avant agrégation)
        questions_par_modele: dict = {}
        for r in resultats:
            modele = r.get("refers_to_model")
            question = r.get("question_content")
            score = r.get("score")
            if modele and question:
                if modele not in questions_par_modele:
                    questions_par_modele[modele] = []
                already = [e["question"] for e in questions_par_modele[modele]]
                if question not in already and len(questions_par_modele[modele]) < 3:
                    questions_par_modele[modele].append({"question": question, "score": score})

        # 3. Calcul du score sémantique de base
        resultats_phase_2 = modeliser_recompense_semantique(resultats)

        # 4. Préparation pour TOPSIS
        # Conversion de la liste envoyée par le front-end en matrice Numpy
        matrice_ahp_np = np.array(request.matrice_ahp)
        
        # Définition stricte des critères utilisés dans cet ordre précis
        noms_criteres = ["score_semantique", "kwh/token", "score_souverainete"]
        libelles_criteres = ["performance_semantique", "energie", "souverainete"]
        poids_ahp = deriver_poids_ahp(matrice_ahp_np)
        poids_criteres = {
            libelle: round(float(poids_ahp[i]), 4)
            for i, libelle in enumerate(libelles_criteres)
        }
        modeles_exclus = calculer_modeles_exclus(
            resultats_semantiques=resultats_phase_2,
            metriques_physiques=metriques_physiques,
            noms_criteres=noms_criteres
        )
        
        # Directions : 1 (Maximiser sémantique), -1 (Minimiser énergie), 1 (Maximiser souveraineté)
        vecteur_directions = [1, -1, 1]

        # 5. Exécution du routage TOPSIS
        classement_final = optimiser_routage_topsis(
            resultats_phase_2=resultats_phase_2,
            metriques_physiques=metriques_physiques,
            matrice_ahp=matrice_ahp_np,
            vecteur_directions=vecteur_directions,
            noms_criteres=noms_criteres
        )

        if not classement_final:
            raise HTTPException(status_code=500, detail="Erreur lors du calcul TOPSIS.")

        # Le grand gagnant est le premier de la liste
        gagnant = classement_final[0]
        classement_detaille = detailler_classement_topsis(
            classement=classement_final,
            resultats_semantiques=resultats_phase_2,
            metriques_physiques=metriques_physiques
        )

        return {
            "prompt": request.prompt,
            "modele_recommande": gagnant[0],
            "score_topsis": gagnant[1],
            "classement_complet": classement_final,
            "classement_detaille": classement_detaille,
            "questions_par_modele": questions_par_modele,
            "poids_criteres": poids_criteres,
            "modeles_exclus": modeles_exclus,
            "infos_analyse": infos_analyse(resultats, seuil)
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))