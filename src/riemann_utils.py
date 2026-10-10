import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.geometry.mean import mean_covariance
from pyriemann.geometry.base import invsqrtm

def estimate_covariances(X: np.ndarray, estimator='lwf') -> np.ndarray:
    """
    Transforme les tenseurs spatio-temporels (N, C, T) en matrices de covariance spatiale (N, C, C).
    Utilise par défaut l'estimateur de Ledoit-Wolf ('lwf') qui est très robuste face au bruit EEG.
    """
    cov_estimator = Covariances(estimator=estimator)
    return cov_estimator.fit_transform(X)

def compute_reference_matrix(covs: np.ndarray) -> np.ndarray:
    """
    Calcule le centre de gravité Riemannien (le 'setup' ou 'point de référence') d'un ensemble de matrices.
    """
    return mean_covariance(covs, metric='riemann')

def apply_euclidean_alignment(covs_target: np.ndarray, ref_matrix: np.ndarray) -> np.ndarray:
    """
    Applique l'Alignement Euclidien (EA) en utilisant une matrice de référence pré-calculée.
    """
    # Calcul de la matrice d'inversion à partir du setup de référence : inverse de la racine carrée de cette moyenne matricielle (M^{-1/2})
    # C'est l'opérateur mathématique qui permet de "décaler" l'espace Riemannien.
    R_invsqrt = invsqrtm(ref_matrix)

    # Recentrage de toutes les matrices de la session cible : : M^{-1/2} * Sigma * M^{-1/2}
    covs_aligned = np.array([R_invsqrt @ c @ R_invsqrt for c in covs_target])

    return covs_aligned

if __name__ == "__main__":
    # Script de test unitaire rapide
    # On crée deux matrices bidons (2 essais, 22 canaux, 513 points) pour simuler la sortie de data_loader.py
    X_mock_s1 = np.random.randn(2, 22, 513)
    X_mock_s2 = np.random.randn(2, 22, 513)
    
    # 1. Compression spatiale
    cov_mock_s1 = estimate_covariances(X_mock_s1)
    cov_mock_s2 = estimate_covariances(X_mock_s2)
    print(f"Dimension après compression : {cov_mock_s1.shape}") # Doit afficher (2, 22, 22)
    
    # 2. Calcul du centre de gravité (Le Setup de référence)
    setup_mock_lundi = compute_reference_matrix(cov_mock_s1)
    print(f"Dimension du setup de référence : {setup_mock_lundi.shape}") # Doit afficher (22, 22)
    
    # 3. Alignement Euclidien
    cov_s1_ea = apply_euclidean_alignment(cov_mock_s1, ref_matrix=setup_mock_lundi)
    cov_s2_ea = apply_euclidean_alignment(cov_mock_s2, ref_matrix=setup_mock_lundi)
    
    print(f"Dimension de S1 après alignement : {cov_s1_ea.shape}") # Doit afficher (2, 22, 22)
    print(f"Dimension de S2 après alignement : {cov_s2_ea.shape}") # Doit afficher (2, 22, 22)