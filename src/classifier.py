import numpy as np
from pyriemann.classification import MDM
from sklearn.metrics import accuracy_score

def train_and_evaluate_baseline(cov_train, y_train, cov_test, y_test):
    """
    Entraîne un classifieur Minimum Distance to Mean (MDM) sur le jeu d'entraînement
    et renvoie sa précision sur le jeu de test.
    """
    # Instanciation de l'algorithme MDM
    mdm = MDM()
    
    # Apprentissage : L'algorithme calcule le centre de gravité des matrices 'left_hand' et le centre des matrices 'right_hand'
    mdm.fit(cov_train, y_train)
    
    # Prédiction : L'algorithme classe les matrices du Mardi selon leur distance aux deux centres calculés précédemment
    y_pred = mdm.predict(cov_test)
    
    # Calcul du score : Pourcentage de prédictions correctes
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def select_top_k_models(ensemble_models, X_calibration, k=11):
    """
    Évalue tous les modèles MDM sur des essais d'étalonnage non labelisés et renvoie les K meilleurs sous forme de comité.
    Sélectionne le modèle dont l'espace latent minimise la distance Riemannienne entre les nouveaux essais et les centres de gravité d'apprentissage.
    """
    scores = []
    
    for i, model in enumerate(ensemble_models):
        distances = model.transform(X_calibration)
        min_distances_per_trial = np.min(distances, axis=1)
        mean_distance = np.mean(min_distances_per_trial)
        
        # On stocke un tuple : (distance_moyenne, le_modele, son_index)
        scores.append((mean_distance, model, i))
        
    # Tri de la liste par distance Riemannienne croissante (du meilleur au pire)
    scores.sort(key=lambda x: x[0])
    
    # Extraction des K premiers éléments
    top_k_models = [item[1] for item in scores[:k]]
    top_k_distances = [item[0] for item in scores[:k]]
    top_k_indices = [item[2] for item in scores[:k]]
        
    return top_k_models, top_k_distances, top_k_indices