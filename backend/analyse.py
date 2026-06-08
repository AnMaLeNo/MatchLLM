from typing import Dict, List, Any, Union, Tuple
import numpy as np



def modeliser_recompense_semantique(
    voisinage_k: List[Dict[str, Any]],
    vecteur_theta: Dict[str, float] = None,
    vecteur_lambda: Dict[str, float] = None,
    alpha: float = 1.0,
    prior_mu: float = None
) -> Dict[str, Dict[str, Union[float, int]]]:
    """
    Calcule la fonction de récompense sémantique, génère le score prédictif S_hat_m,
    et quantifie la cardinalité du support statistique pour chaque modèle de langage.
    
    Args:
        voisinage_k: Liste de dictionnaires (sortie de rechercher_reactions_similaires).
        vecteur_theta: Dictionnaire des pondérations \theta_j pour les signaux positifs.
        vecteur_lambda: Dictionnaire des pénalités \lambda_k pour les signaux négatifs.
        
    Returns:
        Un dictionnaire associant chaque identifiant de modèle à une structure de 
        données contenant le score sémantique agrégé (float) et le volume d'évaluations (int).
    """
    
    # 1. Initialisation des tenseurs de pondération
    if vecteur_theta is None:
        vecteur_theta = {
            "liked": 1.0,
            "useful": 1.5,
            "creative": 1.2,
            "clear_formatting": 0.5 
        }
        
    if vecteur_lambda is None:
        vecteur_lambda = {
            "disliked": 1.0,
            "incorrect": 1.5, 
            "superficial": 1.0,
            "instructions_not_followed": 1.5 
        }

    # Structure d'agrégation incluant le compteur de cardinalité
    donnees_agregation_m = {}

    # 2. Itération sur l'ensemble K
    for interaction in voisinage_k:
        modele_m = interaction.get("refers_to_model")
        if not modele_m:
            continue

        similarite_wi = interaction.get("score", 0.0)
        recompense_ri = 0.0

        # Opérateur de sommation (attributs positifs)
        for signal_j, theta_j in vecteur_theta.items():
            if interaction.get(signal_j) is True:
                recompense_ri += theta_j

        # Opérateur de soustraction (attributs négatifs)
        for signal_k, lambda_k in vecteur_lambda.items():
            if interaction.get(signal_k) is True:
                recompense_ri -= lambda_k

        # 3. Accumulation matricielle et incrémentation du support
        if modele_m not in donnees_agregation_m:
            donnees_agregation_m[modele_m] = {
                "numerateur_somme_ponderee": 0.0,
                "denominateur_somme_poids": 0.0,
                "volume_support": 0
            }

        donnees_agregation_m[modele_m]["numerateur_somme_ponderee"] += similarite_wi * recompense_ri
        donnees_agregation_m[modele_m]["denominateur_somme_poids"] += similarite_wi
        donnees_agregation_m[modele_m]["volume_support"] += 1

    # 4. Inférence probabiliste finale avec lissage bayésien
    resultats_analytiques = {}
    
    if prior_mu is None:
        sum_all_num = sum(m["numerateur_somme_ponderee"] for m in donnees_agregation_m.values())
        sum_all_den = sum(m["denominateur_somme_poids"] for m in donnees_agregation_m.values())
        prior_mu = sum_all_num / sum_all_den if sum_all_den > 0 else 0.0

    for modele_m, metriques in donnees_agregation_m.items():
        # Lissage bayésien : ajout du prior et poids de régularisation
        numerateur_reg = metriques["numerateur_somme_ponderee"] + alpha * prior_mu
        denominateur_reg = metriques["denominateur_somme_poids"] + alpha
        
        score_hat_s = numerateur_reg / denominateur_reg
        volume_support = metriques["volume_support"]
        similarite_moyenne = metriques["denominateur_somme_poids"] / volume_support if volume_support > 0 else 0.0

        if volume_support >= 10 and similarite_moyenne >= 0.55:
            niveau_confiance = "forte"
        elif volume_support >= 3 and similarite_moyenne >= 0.5:
            niveau_confiance = "moyenne"
        else:
            niveau_confiance = "faible"
        
        resultats_analytiques[modele_m] = {
            "score_semantique": round(score_hat_s, 4),
            "volume_support": volume_support,
            "similarite_moyenne": round(similarite_moyenne, 4),
            "niveau_confiance": niveau_confiance
        }

    return resultats_analytiques




def deriver_poids_ahp(matrice_comparaison: np.ndarray) -> np.ndarray:
    """
    Résout le système d'équations de l'Analytic Hierarchy Process (AHP) 
    pour extraire le vecteur de poids normalisé W via le calcul des valeurs propres.
    
    Args:
        matrice_comparaison: Matrice carrée (n x n) des comparaisons par paires.
        
    Returns:
        Vecteur propre principal normalisé (somme = 1).
    """
    n = matrice_comparaison.shape[0]
    
    # 0. Validation des axiomes de la matrice AHP de Saaty
    if matrice_comparaison.shape[0] != matrice_comparaison.shape[1]:
        raise ValueError("La matrice AHP doit être carrée.")
    if np.any(matrice_comparaison <= 0):
        raise ValueError("La matrice AHP doit contenir uniquement des valeurs strictement positives.")
    if not np.allclose(np.diag(matrice_comparaison), 1.0, atol=0.02):
        raise ValueError("La diagonale de la matrice AHP doit être unitaire (a_ii = 1).")
    if not np.allclose(matrice_comparaison * matrice_comparaison.T, np.ones_like(matrice_comparaison), atol=0.02):
        raise ValueError("La matrice AHP doit être réciproque (a_ij = 1 / a_ji).")
    
    # Calcul du spectre de la matrice (valeurs propres et vecteurs propres)
    valeurs_propres, vecteurs_propres = np.linalg.eig(matrice_comparaison)
    
    # Isolation de la valeur propre maximale réelle
    index_max = np.argmax(np.real(valeurs_propres))
    lambda_max = np.real(valeurs_propres[index_max])
    vecteur_propre_principal = np.real(vecteurs_propres[:, index_max])
    
    # Validation du Ratio de Cohérence (Consistency Ratio)
    if n == 2:
        pass # Indice de consistance idéal par définition pour 2x2
    elif n >= 3:
        ci = (lambda_max - n) / (n - 1)
        ri_dict = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        ri = ri_dict.get(n, 1.49) # Approximation pour les n >= 10
        cr = ci / ri
        if cr > 0.10:
            raise ValueError(f"La matrice AHP est incohérente (CR = {cr:.4f} > 0.10)")
    
    # Normalisation L1 du vecteur
    vecteur_poids_w = vecteur_propre_principal / np.sum(vecteur_propre_principal)
    return vecteur_poids_w




def optimiser_routage_topsis(
    resultats_phase_2: Dict[str, Dict[str, Union[float, int]]],
    metriques_physiques: Dict[str, Dict[str, float]],
    matrice_ahp: np.ndarray,
    vecteur_directions: List[int],
    noms_criteres: List[str]
) -> List[Tuple[str, float]]:
    """
    Exécute l'algorithme TOPSIS pour dériver le coefficient de proximité euclidienne C_i.
    
    Args:
        resultats_phase_2: Données issues de `modeliser_recompense_semantique`.
        metriques_physiques: Dictionnaire des coûts matériels par modèle (ex: énergie, latence).
        matrice_ahp: Matrice carrée documentant l'arbitrage utilisateur entre les critères.
        vecteur_directions: Liste d'entiers (1 pour maximiser, -1 pour minimiser) par critère.
        noms_criteres: Ordre strict des attributs pour l'alignement matriciel.
        
    Returns:
        Liste de tuples (identifiant_modele, score_C_i) triée par ordre décroissant de performance.
    """
    
    # 1. Intersection des ensembles d'évaluation
    modeles_candidats = [m for m in resultats_phase_2.keys() if m in metriques_physiques]
    if not modeles_candidats:
        return []

    nombre_modeles = len(modeles_candidats)
    nombre_criteres = len(noms_criteres)
    
    # Filtrer les candidats n'ayant pas toutes les données physiques ou avec des None
    modeles_valides_topsis = []
    for modele in modeles_candidats:
        donnees_semantiques = resultats_phase_2[modele]
        donnees_physiques = metriques_physiques[modele]
        dictionnaire_fusionne = {**donnees_semantiques, **donnees_physiques}
        
        # Vérifie si une valeur est manquante ou None
        invalide = any(dictionnaire_fusionne.get(critere) is None for critere in noms_criteres)
        if not invalide:
            modeles_valides_topsis.append(modele)
            
    modeles_candidats = modeles_valides_topsis
    if not modeles_candidats:
        return []

    nombre_modeles = len(modeles_candidats)
    
    # 2. Étape A : Construction de la matrice de décision X
    matrice_X = np.zeros((nombre_modeles, nombre_criteres))
    
    for i, modele in enumerate(modeles_candidats):
        donnees_semantiques = resultats_phase_2[modele]
        donnees_physiques = metriques_physiques[modele]
        
        # Agrégation dynamique selon l'ordre strict de noms_criteres
        dictionnaire_fusionne = {**donnees_semantiques, **donnees_physiques}
        for j, critere in enumerate(noms_criteres):
            matrice_X[i, j] = float(dictionnaire_fusionne[critere])

    # 3. Étape B : Normalisation Min-Max (r_ij)
    matrice_R = np.zeros_like(matrice_X)
    for j in range(nombre_criteres):
        colonne = matrice_X[:, j]
        min_val = np.min(colonne)
        max_val = np.max(colonne)
        
        # Translation dans [0, 1] et prévention de la division par zéro
        if np.abs(max_val - min_val) < np.finfo(float).eps:
            matrice_R[:, j] = 0.5
        else:
            matrice_R[:, j] = (colonne - min_val) / (max_val - min_val)

    # 4. Étape C : Application des poids AHP (v_ij = W_j * r_ij)
    vecteur_poids_W = deriver_poids_ahp(matrice_ahp)
    matrice_V = matrice_R * vecteur_poids_W

    # 5. Étape D : Détermination des solutions idéales positives (A*) et négatives (A-)
    vecteur_directions_np = np.array(vecteur_directions)
    
    solution_ideale_positive = np.zeros(nombre_criteres)
    solution_ideale_negative = np.zeros(nombre_criteres)
    
    for j in range(nombre_criteres):
        if vecteur_directions_np[j] == 1: # Maximisation (ex: score_semantique)
            solution_ideale_positive[j] = np.max(matrice_V[:, j])
            solution_ideale_negative[j] = np.min(matrice_V[:, j])
        else: # Minimisation (ex: empreinte_energetique)
            solution_ideale_positive[j] = np.min(matrice_V[:, j])
            solution_ideale_negative[j] = np.max(matrice_V[:, j])

    # 6. Étape E : Calcul des distances euclidiennes (D* et D-)
    distances_positives = np.sqrt(np.sum((matrice_V - solution_ideale_positive)**2, axis=1))
    distances_negatives = np.sqrt(np.sum((matrice_V - solution_ideale_negative)**2, axis=1))

    # 7. Étape F : Dérivation du coefficient de proximité C_i
    # L'ajout d'une constante epsilon au dénominateur sécurise les singularités mathématiques.
    scores_closeness_C = np.zeros(nombre_modeles)
    for i in range(nombre_modeles):
        denominateur = distances_positives[i] + distances_negatives[i]
        if denominateur < np.finfo(float).eps:
            scores_closeness_C[i] = 0.5
        else:
            scores_closeness_C[i] = distances_negatives[i] / denominateur

    # Structuration et ordonnancement topologique du classement
    classement_final = [
        (modeles_candidats[i], round(float(scores_closeness_C[i]), 6)) 
        for i in range(nombre_modeles)
    ]
    classement_final.sort(key=lambda x: x[1], reverse=True)

    return classement_final