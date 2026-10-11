# BCI-Telemetry: Adaptation de Domaine Non-Supervisée pour Signaux EEG

## Contexte & Problématique
L'étalonnage des Interfaces Cerveau-Machine (BCI) basées sur l'EEG est entravé par la non-stationnarité des signaux cérébraux. Les modèles d'apprentissage automatique se dégradent drastiquement d'une session à l'autre en raison du *Covariate Shift* induit par la fatigue, l'attention ou le placement du casque. 

## Objectif du Projet
Développer un pipeline de traitement en Python permettant une **Adaptation de Domaine Non-Supervisée** (UDA). L'objectif est d'utiliser les données d'étalonnage d'un Jour A (le simulateur) pour décoder les signaux d'un Jour B (la piste) avec un minimum de recalibrage. 

À l'image d'une écurie de Formule 1, le système pré-calcule des distributions probables ("setups") avant la course à partir des données historiques, puis utilise une inférence ultra-rapide sur les premiers essais non-labelisés d'une nouvelle session pour s'adapter dynamiquement.

## Architecture Technique & Méthodologie
Le pipeline est spécifiquement optimisé pour une **exécution 100% locale sur architecture CPU** (Intel i5). Il s'articule autour des 6 phases suivantes :

1. **Acquisition & Filtrage :** Utilisation du dataset *BCI Competition IV 2a* (Sujets multiples) via MOABB. Filtrage passe-bande (8-32 Hz) pour isoler les rythmes moteurs Mu et Beta, et rééchantillonnage à 128 Hz via MNE-Python.
2. **Châssis Mathématique :** Compression spatiale des séries temporelles en matrices de covariance et application de l'Alignement Euclidien (EA) dans l'Espace Riemannien (`pyRiemann`).
3. **Pilote Automatique (Baseline) :** Évaluation de référence à l'aide d'un classifieur léger *Minimum Distance to Mean* (MDM).
4. **Simulateur F1 (Data Augmentation) :** Modélisation mathématique du *Covariate Shift* pour générer des dizaines de "météos cérébrales" probables et entraîner une flotte d'experts indépendants.
5. **Ancrage Supervisé :** Évaluation des experts sur les tout premiers essais de la session cible (Tour d'installation).
6. **Décodage (Ensemble Learning) :** Vote majoritaire d'un comité restreint (Top-K experts) pour décoder le reste des signaux en temps réel.

## Résultats Actuels & Observations
Au fil du développement, le pipeline a été confronté à la réalité imprévisible des signaux EEG, menant aux conclusions suivantes :

* **Le "Golden Subject" (Sujet 1) :** Les tests initiaux ont montré une excellente précision de base (83.33%). Le simulateur adaptatif a maintenu ce score (82.84%), prouvant la viabilité mathématique du pipeline. Cependant, la stabilité extrême de ce sujet offrait trop peu de *Covariate Shift* pour démontrer l'utilité du système.
* **Le Crash Test et l'Anomalie Matérielle (Sujet 2) :** Sur un sujet sévèrement instable, la précision brute a chuté à 54.86% d'un jour à l'autre. Le simulateur a permis de diagnostiquer que cette dérive n'était pas due à de la fatigue (bruit physiologique lissé), mais à un bruit matériel chaotique (ex: électrode défectueuse) car les dérives entièrement chaotiques tapaient plus souvent dans le mille par rapport aux dérives cohérentes. 
* **L'Échec du Pseudo-Labeling (Adaptation Continue) :** L'activation d'un mode d'apprentissage continu a engendré un grave biais de confirmation. Dans un espace Riemannien instable, le modèle s'auto-validait sur ses propres erreurs avec une certitude absolue, devenant "mathématiquement arrogant". Ce mode a donc été verrouillé.
* **La Solution : L'Ensemble Learning :** La recherche d'un "expert unique parfait" a prouvé ses limites (atteinte d'une asymptote). Le passage à une logique de "Sagesse de la Foule" via un vote d'ensemble a permis de neutraliser le bruit matériel. **L'architecture adaptative a finalement vaincu la Baseline (62.50%) pour se stabiliser à 63.87%**, marquant une certaine victoire d'ingénierie dans des conditions strictes de non-supervision.

## Prochaines Étapes & Vision (Future Work)

Bien que l'architecture actuelle ait permis de surpasser la Baseline et de stabiliser les prédictions par un vote d'ensemble, l'approche purement géométrique a atteint sa limite asymptotique notamment face aux anomalies matérielles sévères. 

La direction que va désormais prendre le projet est l'exploration de nouvelles pistes pour augmenter drastiquement la performance brute de l'architecture. **L'objectif ultime (bien que très ambitieux d'un point de vue clinique) est de réussir à atteindre une précision fiable à hauteur de 80 %**, et ce, même sur les sujets dits "difficiles" présentant un *Covariate Shift* majeur.

Pour briser ce plafond de verre, les prochaines étapes envisageables sont :
- **L'exploration de classifieurs avancés :** Passage à des algorithmes plus complexes projetant les matrices sur l'espace tangent (Tangent Space Classifiers, SVM Riemannien).
- **Le *Transfer Learning* Inter-Sujets :** Exploiter les données d'autres individus pour construire un modèle de référence global et robuste avant même l'acquisition des données d'étalonnage du Jour A.
- **L'intégration d'IA profonde :** Évaluer la pertinence des Réseaux de Neurones Adversariaux (DANN) pour forcer mathématiquement l'effacement du domaine temporel (Session 1 vs Session 2) lors de l'extraction des caractéristiques.

## Installation et Utilisation
```bash
# 1. Cloner le dépôt
git clone [https://github.com/AS-josue/bci-telemetry.git](https://github.com/AS-josue/bci-telemetry.git)
cd bci-telemetry

# 2. Créer et activer l'environnement virtuel
python -m venv venv
# Windows :
.\venv\Scripts\activate
# Linux/Mac :
# source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer le pipeline complet d'évaluation
python main.py