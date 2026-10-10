import numpy as np
import scipy.linalg
from scipy.ndimage import gaussian_filter
from pyriemann.classification import MDM
from joblib import Parallel, delayed

def generate_spatial_noise(n_channels, noise_level=0.05, coherence_radius=1.5, hybrid_ratio=0.0):
    """
    Génère une matrice de transformation spatiale (déformation) hybride pour la Data Augmentation Riemannienne contrôlée par le ratio hybride :
    1.0 = 100% Cohérent (Dérive Physiologique globale)
    0.0 = 100% Chaotique (Anomalie Matérielle locale)
    """

    # 1. Couche Matérielle (anomalies locales de type électrode débranchée)
    W_raw = np.random.randn(n_channels, n_channels)

    # 2. Couche Physiologique (dérive spatiale lissée de type fatigue)
    # Les électrodes voisines sont désormais corrélées (plus coherence_radius est grand, plus la déformation est large).
    W_smooth = gaussian_filter(W_raw, sigma=coherence_radius)

    # 3. Normalisation des énergies pour égaliser le poids géométrique des deux matrices
    W_raw = W_raw / np.std(W_raw)
    W_smooth = W_smooth / np.std(W_smooth)

    # 4. Fusion Hybride
    W = (hybrid_ratio * W_smooth) + ((1 - hybrid_ratio) * W_raw)

    # 5. Restauration de l'amplitude pour que noise_level contrôle exactement la force de déformation
    W = W / np.std(W)

    # 6. Création de la matrice anti-symétrique et projection sur la variété géométrique orthogonale
    # L'intensité de la rotation est contrôlée par noise_level
    W = (W - W.T) / 2
    R = scipy.linalg.expm(noise_level * W)

    return R

def augment_covariances(covs, max_noise_level=0.25, step_noise=0.01, setups_per_level=5):
    # Génère une flotte d'experts en croisant l'intensité, le profil d'anomalie et la direction spatiale.

    setups = []
    n_channels = covs.shape[1]
    
    # 1. L'ingénieur de base : donnée pure (0% de dérive)
    setups.append(covs.copy())
    
    # 2. Grille d'intensités avec balayage automatique (ex: 0.01, 0.02... jusqu'à max_noise_level)
    # On ajoute très légèrement step_noise à la borne max pour s'assurer qu'elle est incluse par numpy
    intensites = np.arange(step_noise, max_noise_level + (step_noise/2), step_noise)
    
    # 3. Les trois profils d'anomalies
    profils_anomalie = [
        # 1.0 # Physiologique (Lissé)
        0.0 # Matériel (Chaotique)
        # 0.5  # Mixte
    ]
    
    # 4. Génération multidimensionnelle
    for current_noise in intensites:
        for profil in profils_anomalie:
            # N directions différentes (setups_per_level) pour cette combinaison spécifique (intensité + profil)
            for _ in range(setups_per_level):
                R = generate_spatial_noise(n_channels, current_noise, hybrid_ratio=profil)
                
                # Vectorisation : NumPy applique R sur l'ensemble des N matrices simultanément
                augmented_covs = R @ covs @ R.T 
                setups.append(augmented_covs)
                
    return setups

"""
def generate_spatial_noise(n_channels, noise_level=0.05):
    # Génère une matrice de transformation spatiale proche de l'Identité.
    # Simule un léger déplacement du casque ou un changement d'impédance de la peau.
    
    # Création d'une matrice aléatoire anti-symétrique
    W = np.random.randn(n_channels, n_channels)

    # Exponentielle matricielle pour obtenir une véritable matrice de rotation (orthogonale)
    # L'intensité de la rotation est contrôlée par noise_level
    W = (W - W.T) / 2
    R = scipy.linalg.expm(noise_level * W)
    return R
"""

def train_single_mdm(X_setup, y_train):
    # Fonction isolée pour l'entraînement d'un seul expert.
    mdm = MDM()
    mdm.fit(X_setup, y_train)
    return mdm

def train_ensemble(setups_X, y_train):
    # Entraîne l'ensemble des classifieurs MDM en parallèle.
    # n_jobs=-1 ordonne d'utiliser 100% des cœurs CPU disponibles.

    # Distribution de la charge sur tous les cœurs du processeur
    ensemble_models = Parallel(n_jobs=-1)(
        delayed(train_single_mdm)(X_setup, y_train) for X_setup in setups_X
    )
        
    return ensemble_models