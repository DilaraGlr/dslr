import sys
import pandas as pd
import numpy as np
import json


def load_weights(filepath='weights.json'):
    """Charge les poids et paramètres depuis le fichier JSON"""
    try:
        # Ouvrir et lire le fichier JSON
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier {filepath} n'existe pas")
        print("   Lance d'abord logreg_train.py pour créer les poids")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        sys.exit(1)

    # Extraire les 4 éléments du dictionnaire
    feature_names = data['feature_names']

    # Convertir means et stds en numpy arrays pour les calculs
    means = np.array(data['means'])
    stds = np.array(data['stds'])

    # Laisser weights en listes pour l'instant (converti plus tard dans
    # predict)
    weights = data['weights']

    print(f"✅ Poids chargés depuis : {filepath}")
    print(f"   Features : {len(feature_names)}")
    print(f"   Maisons  : {list(weights.keys())}")

    return feature_names, means, stds, weights


def load_data(filepath, feature_names):
    """Charge le test set et extrait les features"""
    # Charger le CSV
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Fichier chargé : {filepath}")
        print(f"   {len(df)} étudiants à prédire")
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier {filepath} n'existe pas")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        sys.exit(1)

    # Vérifier que toutes les features existent
    for feat in feature_names:
        if feat not in df.columns:
            print(f"❌ Erreur : Feature '{feat}' introuvable dans le test set")
            sys.exit(1)

    # Extraire les bonnes colonnes et convertir en numpy array
    X = df[feature_names].values

    print(f"✅ Features extraites : {len(feature_names)}")
    print(f"   Shape de X : {X.shape}")

    return X


def handle_missing_values(X, means):
    """
    Remplace les NaN par les moyennes du train set

    Args:
        X: numpy array (m, n) - peut contenir des NaN
        means: numpy array (n,) - moyennes de chaque feature (du train)

    Returns:
        X_clean: numpy array (m, n) - sans NaN

    Note:
        Utilise les moyennes du TRAIN SET (pas du test set!)
        C'est crucial pour éviter le data leakage
    """
    X_clean = X.copy()
    m, n = X.shape
    total_nans = 0

    # Pour chaque colonne
    for j in range(n):
        # Trouver les NaN dans cette colonne
        nan_mask = np.isnan(X_clean[:, j])
        nan_count = np.sum(nan_mask)

        if nan_count > 0:
            # Remplacer par la moyenne du train
            X_clean[nan_mask, j] = means[j]
            total_nans += nan_count

    print(f"✅ Valeurs manquantes traitées : {total_nans} NaN remplacés")
    if total_nans > 0:
        print("   Stratégie : moyennes du train set")

    return X_clean


def standardize(X, means, stds):
    """
    Standardise avec les paramètres du train set

    Args:
        X: numpy array (m, n)
        means: numpy array (n,) - moyennes du train
        stds: numpy array (n,) - écarts-types du train

    Returns:
        X_norm: numpy array (m, n) - données standardisées

    Note:
        Utilise les MÊMES moyennes et écarts-types que le train!
        Formule : X_norm = (X - means) / stds
    """
    X_norm = X.copy()

    # Pour chaque colonne
    for j in range(X.shape[1]):
        if stds[j] > 0:
            X_norm[:, j] = (X[:, j] - means[j]) / stds[j]
        else:
            # Si std = 0 dans le train, juste centrer
            X_norm[:, j] = X[:, j] - means[j]

    print("✅ Standardisation appliquée (paramètres du train)")
    print(f"   X_norm : min={X_norm.min():.2f}, max={X_norm.max():.2f}")

    return X_norm


def add_intercept(X):
    """
    Ajoute une colonne de 1 au début

    Args:
        X: numpy array (m, n)

    Returns:
        X_with_intercept: numpy array (m, n+1)
    """
    m = X.shape[0]
    ones = np.ones(m)
    X_with_intercept = np.c_[ones, X]

    print("✅ Colonne d'intercept ajoutée")
    print(f"   Shape : {X.shape} → {X_with_intercept.shape}")

    return X_with_intercept


def sigmoid(z):
    """Fonction sigmoïde pour la régression logistique"""
    # Clipper z pour éviter l'overflow (e^1000 = trop grand)
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def predict(X, weights):
    """
    Prédit les maisons pour les données X

    Args:
        X: numpy array (m, n) - données préparées (avec intercept)
        weights: dict - poids de chaque maison

    Returns:
        predictions: list - maison prédite pour chaque étudiant
    """
    # Convertir les poids en numpy array pour le calcul
    house_names = list(weights.keys())
    weight_matrix = np.array([weights[house] for house in house_names])
    # shape (num_houses, n)

    # Calculer les scores pour chaque maison
    scores = X @ weight_matrix.T  # shape (m, num_houses)

    # Appliquer la sigmoïde pour obtenir des probabilités
    probabilities = sigmoid(scores)

    # Prédire la maison avec la probabilité la plus élevée
    predicted_indices = np.argmax(probabilities, axis=1)
    predictions = [house_names[idx] for idx in predicted_indices]

    return predictions


def save_predictions(predictions, output_filepath='houses.csv'):
    """
    Sauvegarde les prédictions dans un fichier CSV

    Format requis par le sujet :
        Index,Hogwarts House
        0,Gryffindor
        1,Hufflepuff
        ...
    """
    # Créer un DataFrame avec le format exact demandé
    df_output = pd.DataFrame({
        'Index': range(len(predictions)),
        'Hogwarts House': predictions
    })

    # Sauvegarder avec index=False pour éviter une colonne index supplémentaire
    df_output.to_csv(output_filepath, index=False)

    print(f"✅ Prédictions sauvegardées dans : {output_filepath}")
    print("   Format : Index,Hogwarts House")
    print(f"   {len(predictions)} prédictions")


def main():
    """Point d'entrée principal"""
    # Récupérer les fichiers
    filepath_test = sys.argv[1] if len(sys.argv) > 1 \
        else 'data/dataset_test.csv'
    filepath_weights = sys.argv[2] if len(sys.argv) > 2 \
        else 'weights.json'

    print("=" * 60)
    print("⚡ POUDLARD - PRÉDICTION DES MAISONS ⚡")
    print("=" * 60)
    print()

    # Étape 1 : Charger les poids
    print("📂 ÉTAPE 1 : Chargement des poids")
    print("-" * 60)
    feature_names, means, stds, weights = load_weights(filepath_weights)
    print()

    # Étape 2 : Charger le test set
    print("📂 ÉTAPE 2 : Chargement du test set")
    print("-" * 60)
    X = load_data(filepath_test, feature_names)
    print()

    # Étape 3 : Préparer X (même pipeline que le train)
    print("🔧 ÉTAPE 3 : Préparation des données")
    print("-" * 60)

    # 3a. Gérer les NaN avec les moyennes du train
    X = handle_missing_values(X, means)

    # 3b. Standardiser avec les paramètres du train
    X = standardize(X, means, stds)

    # 3c. Ajouter intercept
    X = add_intercept(X)
    print()

    # Étape 4 : Prédire les maisons
    print("🎯 ÉTAPE 4 : Prédiction des maisons")
    print("-" * 60)
    predictions = predict(X, weights)
    print(f"✅ Prédictions générées : {len(predictions)} étudiants")
    print(f"   Exemples : {predictions[:5]}")
    print()

    # Étape 5 : Sauvegarder les prédictions
    print("💾 ÉTAPE 5 : Sauvegarde des prédictions")
    print("-" * 60)
    save_predictions(predictions, 'houses.csv')
    print()
    print("=" * 60)
    print("⚡ PRÉDICTION TERMINÉE !")
    print("=" * 60)


if __name__ == '__main__':
    main()
