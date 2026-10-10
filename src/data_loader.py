# Phase 1 : Extraction, filtrage et séparation des sessions EEG.

import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
import warnings

# Ignorer les avertissements liés aux dépréciations internes de MNE/MOABB
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def load_and_prepare_data(subject: int = 1):
    """
    Télécharge, filtre et sépare les données EEG d'un sujet spécifique.
    
    Args:
        subject (int): Le numéro du sujet (1 par défaut).
        
    Returns:
        tuple: (X_s1, y_s1, X_s2, y_s2)
            X_s1, y_s1 : Données et labels de la Session 1 (Lundi / Entraînement)
            X_s2, y_s2 : Données et labels de la Session 2 (Mardi / Test)
    """
    print(f"Téléchargement et préparation des données pour le Sujet {subject}...")
    
    # 1. Instanciation du dataset
    dataset = BNCI2014_001()
    
    # 2. Configuration du paradigme (Filtrage 8-32Hz et Rééchantillonnage 128Hz)
    paradigm = LeftRightImagery(fmin=8, fmax=32, resample=128)
    
    # 3. Extraction brute (X, y et métadonnées)
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])  # type: ignore
    
    # 4. Création des masques pour séparer le Lundi du Mardi
    # La colonne 'session' contient '0train' pour la session 1 et '1test' pour la session 2
    mask_s1 = meta['session'] == '0train'
    mask_s2 = meta['session'] == '1test'
    
    # 5. Application des masques
    X_s1 = X[mask_s1]
    y_s1 = y[mask_s1]
    
    X_s2 = X[mask_s2]
    y_s2 = y[mask_s2]
    
    return X_s1, y_s1, X_s2, y_s2

if __name__ == "__main__":
    # Test local du script
    X_s1, y_s1, X_s2, y_s2 = load_and_prepare_data(subject=1) # type: ignore
    
    print("\n--- Validation des dimensions (N, C, T) ---")
    print(f"Session 1 (Lundi) - X_s1 : {X_s1.shape} | Labels y_s1 : {y_s1.shape}")
    print(f"Session 2 (Mardi) - X_s2 : {X_s2.shape} | Labels y_s2 : {y_s2.shape}")
    
    # Vérification des classes
    classes_s1 = np.unique(y_s1)
    print(f"Classes détectées : {classes_s1}")

    # --- Visualisation Graphique ---
    import matplotlib.pyplot as plt
    import numpy as np

    # On récupère dynamiquement le nombre de points temporels (ici 513)
    n_points = X_s1.shape[2]
    freq_echantillonnage = 128.0 # 128 Hz

    # On calcule la durée réelle de l'essai en secondes (513 / 128 ≈ 4.01s)
    duree_secondes = n_points / freq_echantillonnage

    # Création de l'axe temporel dynamique
    temps = np.linspace(0, duree_secondes, n_points)

    # Sélection d'une seule électrode (ex: l'index 7, qui correspondrait à C3/C4 selon la topologie)
    canal_idx = 7 

    # Séparation logique des essais selon leurs labels
    essais_gauche = X_s1[y_s1 == 'left_hand']
    essais_droite = X_s1[y_s1 == 'right_hand']

    # On limite à 3 ou 4 essais par graphe pour garantir la lisibilité
    nb_essais = 3

    # Palettes de couleurs contrastées pour bien dissocier chaque essai
    couleurs_droite = ['#ff7f0e', '#9467bd', '#2ca02c'] # Orange, Violet, Vert
    couleurs_gauche = ['#17becf', '#e377c2', '#bcbd22'] # Cyan, Rose, Moutarde

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # 1. Graphe Main Droite (Couleurs contrastées)
    for i in range(nb_essais):
        axes[0].plot(temps, essais_droite[i, canal_idx, :], 
                    color=couleurs_droite[i % len(couleurs_droite)], 
                    linewidth=1.5, label=f'Essai {i+1} (Droit)')
    axes[0].set_title("Superposition intra-classe : Main Droite (right_hand)", fontweight='bold')
    axes[0].set_ylabel("Amplitude (µV)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 2. Graphe Main Gauche (Couleurs contrastées)
    for i in range(nb_essais):
        axes[1].plot(temps, essais_gauche[i, canal_idx, :], 
                    color=couleurs_gauche[i % len(couleurs_gauche)], 
                    linewidth=1.5, label=f'Essai {i+1} (Gauche)')
    axes[1].set_title("Superposition intra-classe : Main Gauche (left_hand)", fontweight='bold')
    axes[1].set_ylabel("Amplitude (µV)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # 3. Graphe Comparatif Inter-classes (Maintien du Rouge/Bleu classique pour l'opposition)
    for i in range(nb_essais):
        # Gestion des labels pour éviter les doublons dans la légende
        label_d = 'Main Droite' if i == 0 else ""
        label_g = 'Main Gauche' if i == 0 else ""
        
        axes[2].plot(temps, essais_droite[i, canal_idx, :], color='firebrick', alpha=0.5, label=label_d)
        axes[2].plot(temps, essais_gauche[i, canal_idx, :], color='steelblue', alpha=0.5, label=label_g)

    axes[2].set_title("Comparaison inter-classes : Droite vs Gauche", fontweight='bold')
    axes[2].set_xlabel("Temps (secondes)")
    axes[2].set_ylabel("Amplitude (µV)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()