# Matcher d'IA — Routeur Intelligent

**"On matche vos prompts avec l'IA qui leur correspond : le système qui unit chaque requête à son LLM idéal."**

> 🌐 Démo en ligne : [sbo42.fr/MatchLLM](https://sbo42.fr/MatchLLM/)

Cette application est un système d'aide à la décision conçu pour orienter chaque utilisateur vers le modèle de langage (LLM) le plus adapté à son besoin spécifique. En exploitant les données massives de la plateforme **Compar:IA**, l'outil ne se contente pas de statistiques globales : il analyse sémantiquement votre requête pour trouver quel modèle a historiquement le mieux performé sur des sujets similaires.

---

## ⚙️ Architecture Analytique : Évaluation à Double Niveau

L'application repose sur un moteur d'analyse qui combine le retour d'expérience des utilisateurs et l'optimisation multi-critères.

### 1. Analyse de la Récompense Sémantique

Ce module évalue la performance des modèles en se basant sur la similarité des intentions.

- **Recherche Vectorielle :** Utilisation du modèle d'embedding `BAAI/bge-m3` et de la base `Qdrant` pour retrouver les conversations historiques les plus proches de votre prompt.
- **Pondération des Signaux Utilisateurs :** Le score est calculé à partir des réactions réelles des utilisateurs de Compar:IA, récupérées via le fichier `reactions.parquet`.
    - **Signaux Positifs :** Les tags comme `useful` (utile) ou `creative` (créatif) augmentent la note du modèle.
    - **Signaux Négatifs :** Les tags comme `incorrect` ou `instructions_not_followed` (consignes non suivies) la pénalisent lourdement.
- **Inférence et Lissage :** Pour garantir la fiabilité statistique, le système utilise un lissage bayésien. Cela permet d'ajuster le score des modèles ayant peu d'historique en les comparant à une moyenne globale (Prior), évitant ainsi les résultats extrêmes basés sur trop peu d'échantillons.

### 2. Arbitrage Multi-Critères (AHP + TOPSIS)

Le système permet également d'équilibrer la performance avec des contraintes physiques et éthiques.

- **AHP (Analytic Hierarchy Process) :** L'utilisateur définit l'importance relative de la **Performance sémantique**, de l'**Énergie** (kWh/token) et de la **Souveraineté**. Le code traduit ces préférences en poids mathématiques cohérents.
- **TOPSIS :** Cet algorithme classe les modèles en calculant leur distance par rapport à une "solution idéale" (le modèle parfait sur tous les critères) et une "solution anti-idéale" (le pire compromis).

---

## Données d'Indexation : Sens & Feedback

Pour fonctionner, le système doit fusionner deux flux de données lors de la phase d'indexation (`index_corpus`) :

1. **`base_vectorielle_reactions_question_content.parquet` (Le "Sens")** : Contient les **embeddings** (vecteurs) des questions. C'est ce qui permet au système de "comprendre" sémantiquement le texte.
2. **`reactions.parquet` (Le "Feedback")** : Contient les **données brutes** de la plateforme Compar:IA (nom du modèle, tags `liked`, `useful`, `incorrect`, etc.). Ce sont ces étiquettes laissées par les humains qui permettent de noter les modèles.

### ⚡ Note sur la Base Vectorielle

La génération des embeddings (la transformation du texte en vecteurs) est une étape très lourde qui dépend fortement du matériel (GPU/CPU) et peut prendre plusieurs heures. Pour gagner du temps et éviter une consommation électrique inutile pour recalculer des données statiques, **nous fournissons la base vectorielle déjà générée**.

> 📂 **Téléchargement :** Les fichiers (Vecteurs et Réactions) sont disponibles sur ce [Drive Google](https://drive.google.com/drive/folders/1usXZcvWKPYTvAiUEq6vxNn_jwlOKY8NM?usp=share_link&authuser=1).
> 

---

## Génération Manuelle des Données (Optionnel)

Bien que nous fournissions la base pré-calculée, vous pouvez régénérer l'ensemble du pipeline de données (pour mettre à jour les statistiques ou changer le modèle d'embedding) en utilisant les outils situés dans `./backend/tools/`.

### Étapes de génération :

**1. Préparation** Placez vos fichiers bruts (`reactions.parquet` et `conversations.parquet`) dans le dossier `backend/database/`.

**2. Calcul des métriques physiques** Ce script utilise `conversations.parquet` pour calculer la consommation énergétique moyenne (kWh/token) et la souveraineté par modèle.

```bash
python backend/tools/extract_model_stats_to_json.py
```

- **Sortie** : `backend/metriques_physiques.json`.

**3. Génération des vecteurs** Exécutez le script de vectorisation. Il va transformer les textes des questions (`reactions.parquet`) en coordonnées mathématiques.

```bash
python backend/tools/embeddings.py
```
- **Fonctionnement :** Le script génère de multiples petits fichiers au fur et à mesure. Cette approche permet d'arrêter le script à tout moment sans perdre le travail déjà effectué.
- **Sécurité :** En cas de coupure (matérielle ou manuelle), le script reprendra automatiquement là où il s'est arrêté lors de la prochaine exécution.
- **Note technique :** Nécessite `sentence-transformers` et idéalement un GPU pour des performances acceptables.

**4. Fusion et finalisation de la base** Ce script assemble les vecteurs générés avec les métadonnées de feedback pour créer le fichier d'indexation.

```bash
python backend/tools/fusion_embeddings.py
```

- **Sortie** : `base_vectorielle_complete.parquet`.

> 💡 **Note importante :** Pour que l'application puisse utiliser ce fichier, renommez-le en `base_vectorielle_reactions_question_content.parquet` et déplacez-le dans le dossier `backend/base_vectorielle/`.
> 

---

## Installation

1. **Démarrer l'infrastructure :**

```bash
docker-compose up -d --build
```
    
2. **Initialiser la base de données :** Placez les fichiers `.parquet` téléchargés dans les dossiers `database` et `base_vectorielle` respectifs du backend. L'indexation dans Qdrant s'effectue automatiquement au lancement du serveur.

3. **Accéder à l'application :** Ouvrez votre navigateur et rendez-vous sur l'adresse `http://localhost:8078`.