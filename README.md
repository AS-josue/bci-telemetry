# BCI-Telemetry

## Contexte
L'étalonnage des Interfaces Cerveau-Machine (BCI) basées sur l'EEG est entravé par la non-stationnarité des signaux cérébraux. Les modèles d'apprentissage automatique se dégradent drastiquement d'une session à l'autre en raison du *Covariate Shift* induit par la fatigue, l'attention ou le placement du casque.

## Objectif du Projet
Développer un pipeline de traitement en Python permettant une **Adaptation de Domaine Non-Supervisée**. L'objectif est de pré-calculer des distributions probables ("setups") à partir des données d'étalonnage initiales, puis d'utiliser une inférence rapide sur les premiers essais d'une nouvelle session pour s'adapter sans recalibrage complet.

## Architecture Technique
- **Acquisition :** Dataset BCI Competition IV 2a (via MOABB).
- **Traitement du signal :** MNE-Python (filtrage spatial, extraction d'Epochs).
- **Modélisation & ML :** Géométrie Riemannienne (`pyRiemann`) et `scikit-learn`. 
- **Contrainte :** Pipeline optimisé pour une exécution 100% locale sur architecture CPU.

## Installation et Utilisation
```bash
git clone [https://github.com/AS-josue/bci-telemetry.git](https://github.com/AS-josue/bci-telemetry.git)
cd bci-telemetry
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt