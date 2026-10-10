from sklearn.metrics import accuracy_score
from collections import Counter
import numpy as np
# np.random.seed(42) # Fige la constellation des dérives

# Importation de nos propres modules
from src.data_loader import load_and_prepare_data
from src.riemann_utils import estimate_covariances, compute_reference_matrix, apply_euclidean_alignment
from src.classifier import train_and_evaluate_baseline, select_top_k_models
from src.augmentation import augment_covariances, train_ensemble

def main():
    print("=== Démarrage du Pipeline BCI-Telemetry ===")
    
    # ---------------------------------------------------------
    # PHASE 1 : Télémétrie de Base (Acquisition)
    # ---------------------------------------------------------
    print("\n--- PHASE 1 : Chargement des données ---")
    # On récupère les 4 variables (S1=Lundi, S2=Mardi) pour le sujet
    X_s1, y_s1, X_s2, y_s2 = load_and_prepare_data(subject=1) # type: ignore

    """
    print(f"Dimensions brutes S1 (Lundi) : {X_s1.shape}")
    print(f"Dimensions brutes S2 (Mardi) : {X_s2.shape}")
    """
    
    # ---------------------------------------------------------
    # PHASE 2 : Châssis Mathématique (Espace Riemannien)
    # ---------------------------------------------------------
    print("\n--- PHASE 2 : Transformation Riemannienne et Alignement ---")

    # Étape A : Compression Temporelle -> Spatiale
    cov_s1 = estimate_covariances(X_s1)
    cov_s2 = estimate_covariances(X_s2)

    # Étape B : Extraction des Setups
    setup_lundi = compute_reference_matrix(cov_s1)
    setup_mardi = compute_reference_matrix(cov_s2)
    
    # Étape C : Alignement Euclidien (EA) : Chaque jour est recentré sur lui-même pour annuler sa propre dérive
    cov_s1_ea = apply_euclidean_alignment(covs_target=cov_s1, ref_matrix=setup_lundi)
    cov_s2_ea = apply_euclidean_alignment(covs_target=cov_s2, ref_matrix=setup_mardi)

    """
    print(f"Dimension de S1 après compression et alignement : {cov_s1_ea.shape}")
    print(f"Dimension de S2 après compression et alignement : {cov_s2_ea.shape}")
    """

    # ---------------------------------------------------------
    # PHASE 3 : Établissement de la Référence (Baseline)
    # ---------------------------------------------------------
    print("\n--- PHASE 3 : Évaluation du Pilote Automatique (MDM) ---")
    
    # Test 1 : Sans correction mathématique (Covariate Shift complet)
    score_brut = train_and_evaluate_baseline(cov_train=cov_s1, y_train=y_s1, cov_test=cov_s2, y_test=y_s2)
    
    # Test 2 : Avec l'Alignement Euclidien de la Phase 2
    score_ea = train_and_evaluate_baseline(cov_train=cov_s1_ea, y_train=y_s1, cov_test=cov_s2_ea, y_test=y_s2)

    print(f"Précision sur le Mardi SANS alignement : {score_brut * 100:.2f} %")
    """
    print(f"Précision sur le Mardi AVEC alignement (La Vraie Baseline) : {score_ea * 100:.2f} %")
    print(f"-> Gain apporté par l'algèbre de la Phase 2 : {((score_ea - score_brut) * 100):.2f} points")
    """

    # ---------------------------------------------------------
    # PHASE 4 & 5 : Ancrage Supervisé et Adaptation Dynamique (Temps Réel)
    # ---------------------------------------------------------
    print("\n--- PHASE 4 & 5 : Ancrage Supervisé et Adaptation Dynamique ---")

    n_essais_calibration = 10 # Vrais labels (Le tour de chauffe)
    n_essais_validation = 5   # Sans labels (Évaluation géométrique de la trajectoire)

    # 1. Extraction Temps Réel : On isole les données BRUTES chronologiquement
    # Tour de chauffe (0 à n_essais_calibration)
    cov_mardi_calib_brut = cov_s2[:n_essais_calibration]
    y_mardi_calib = y_s2[:n_essais_calibration]

    # Validation (10 à 15)
    cov_mardi_validation_brut = cov_s2[n_essais_calibration : n_essais_calibration + n_essais_validation]

    # Reste de la course (15 à la fin)
    cov_mardi_course_brut = cov_s2[n_essais_calibration + n_essais_validation:]
    y_course = y_s2[n_essais_calibration + n_essais_validation:]

    # 2. Traitement à la volée : Alignement du Tour de Chauffe
    # On utilise setup_lundi car c'est la seule référence mathématique connue à cet instant
    X_mardi_calib_aligne = apply_euclidean_alignment(cov_mardi_calib_brut, setup_lundi)

    # 3. L'Ancrage Supervisé (Fusion)
    print(f"Recalibrage des centres géométriques sur les {n_essais_calibration} essais reçus...")
    X_combines = np.concatenate((cov_s1_ea, X_mardi_calib_aligne), axis=0)
    y_combines = np.concatenate((y_s1, y_mardi_calib), axis=0)

    # 4. Le simulateur inspiré de la F1 : Déploiement à partir du nouvel ancrage
    plafond_derive = 0.25      # On simule jusqu'à 25% de décalage (Possibilité de pousser à 40% pour capturer les dérives extrêmes)
    pas_derive = 0.005          # Un palier tous les 0.5% (Réduit à 0.5% pour une granularité chirurgicale)
    directions_par_palier = 15  # setups (directions spatiales) pour chaque cas

    print(f"Déploiement de l'équipe d'ingénieurs sur {plafond_derive * 100:.0f}% de dérive simulée...")
    setups_mardi = augment_covariances(X_combines, max_noise_level=plafond_derive, step_noise=pas_derive, setups_per_level=directions_par_palier)
    equipe_ingenieurs = train_ensemble(setups_mardi, y_combines)
    print(f"{len(setups_mardi)} scénarios de dérive ont été déployés et les experts MDM entraînés en arrière-plan.")

    # 5. Évaluation Géométrique sur la Validation
    # On reçoit 5 nouveaux essais bruts sans labels, on les aligne instantanément
    X_mardi_validation_aligne = apply_euclidean_alignment(cov_mardi_validation_brut, setup_lundi)
    
    k_experts = 11 # Un nombre impair est préférable pour éviter les égalités parfaites

    print(f"Évaluation de l'équipe sur les {n_essais_validation} essais de validation...")
    
    top_modeles, top_distances, top_indices = select_top_k_models(
        equipe_ingenieurs, X_mardi_validation_aligne, k=k_experts
    )
    
    print(f"-> Un comité de {k_experts} experts a été verrouillé.")
    print(f"-> Le leader du comité est le Setup #{top_indices[0]} (Distance: {top_distances[0]:.4f})")

    # 6. Inférence Finale (Sagesse de la Foule)
    print("\n--- PHASE 6 : Décodage de la Course par Vote Majoritaire ---")

    # --- PARAMÈTRES DE CONTRÔLE ---
    continuous_adaptation = False # True = Pseudo-Labeling Continu | False = Mode Statique (Pas de mise à jour)
    taille_bloc = 5 # Mise à jour géométrique tous les 5 essais (si continuous_adaptation = True)
    seuil_confiance = 0.65 
    
    y_pred_course = []
    
    if not continuous_adaptation:
        print("Mode Statique : Prédiction de la course en un seul bloc (Vote Majoritaire)...")
        
        # On aligne le reste de la course au fur et à mesure (simulé ici en bloc)
        X_mardi_course_aligne = apply_euclidean_alignment(cov_mardi_course_brut, setup_lundi)
        
        # Récupération des prédictions de chaque membre du comité
        toutes_predictions = []

        for i, modele in enumerate(top_modeles):
            y_pred_expert = modele.predict(X_mardi_course_aligne)
            toutes_predictions.append(y_pred_expert)

        # Convertir en matrice NumPy (shape: [11 experts, N essais de la course])    
        toutes_predictions = np.array(toutes_predictions)
        
        # Calcul du vote majoritaire pour chaque colonne (chaque essai) manuellement avec Counter (compatible texte)
        for col in range(toutes_predictions.shape[1]):
            votes = toutes_predictions[:, col]
            vainqueur = Counter(votes).most_common(1)[0][0]
            y_pred_course.append(vainqueur)
            
    else:
        print(f"Mode Continu : Activation du Pseudo-Labeling Sélectif par Comité (Seuil: {seuil_confiance})...")

        # Initialisation des historiques individuels pour CHAQUE expert du comité
        X_courants = [setups_mardi[idx].copy() for idx in top_indices]
        y_courants = [y_combines.copy() for _ in top_indices]
        
        # Traitement séquentiel (Simulation du flux Temps Réel)
        for i in range(0, len(cov_mardi_course_brut), taille_bloc):
            # Extraction et alignement du bloc en cours
            bloc_brut = cov_mardi_course_brut[i : i + taille_bloc]
            X_bloc_aligne = apply_euclidean_alignment(bloc_brut, setup_lundi)
            
            # 1. Obtenir les prédictions de tous les experts pour ce bloc
            predictions_bloc_experts = []
            for modele in top_modeles:
                predictions_bloc_experts.append(modele.predict(X_bloc_aligne))
            
            predictions_bloc_experts = np.array(predictions_bloc_experts)
            
            # 2. Vote majoritaire manuel pour le bloc
            y_bloc_vote = []
            for col in range(predictions_bloc_experts.shape[1]):
                votes = predictions_bloc_experts[:, col]
                vainqueur = Counter(votes).most_common(1)[0][0]
                y_bloc_vote.append(vainqueur)
                
            y_pred_course.extend(y_bloc_vote)
            
            # 3. Pseudo-Labeling Individuel : Chaque expert s'évalue et s'adapte indépendamment
            for k, modele in enumerate(top_modeles):
                probas = modele.predict_proba(X_bloc_aligne)
                y_pred_expert = predictions_bloc_experts[k] # La prédiction brute de cet expert précis
                
                X_valides = []
                y_valides = []
                
                # Filtrage probabiliste pour cet expert
                for j in range(len(bloc_brut)):
                    confiance_max = np.max(probas[j]) 
                    
                    if confiance_max > seuil_confiance:
                        X_valides.append(X_bloc_aligne[j])
                        y_valides.append(y_pred_expert[j])
                
                # 4. Mise à jour de l'expert courant (uniquement si des essais ont passé son propre filtre)
                if len(X_valides) > 0:
                    X_courants[k] = np.concatenate((X_courants[k], np.array(X_valides)), axis=0)
                    y_courants[k] = np.concatenate((y_courants[k], np.array(y_valides)), axis=0)

                    # Ré-entraînement instantané de l'expert K
                    modele.fit(X_courants[k], y_courants[k])

    # Conversion finale pour scikit-learn
    y_pred_course = np.array(y_pred_course)
    score_dynamique = accuracy_score(y_course, y_pred_course)

    print(f"\n=== VERDICT FINAL ===")
    print(f"Précision Baseline (Phase 3) : {score_ea * 100:.2f} %")
    print(f"Précision d'Ensemble (Vote à {k_experts}) : {score_dynamique * 100:.2f} %")
    
    gain_final = (score_dynamique - score_ea) * 100
    if gain_final > 0:
        print(f"🏆 Succès : L'architecture a gagné +{gain_final:.2f} points de précision !")
    else:
        print(f"⚠️ Le système n'a pas fait mieux que le modèle moyen cette fois-ci.")

    print("\n=== Fin de l'exécution ===")

if __name__ == "__main__":
    main()