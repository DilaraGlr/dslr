import sys
import pandas as pd
import numpy as np
import json


def load_data(filepath, selected_features=None):
    """Charge les données depuis le CSV et extrait X et y"""
    # 1. Charger le CSV
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Fichier chargé : {filepath}")
        print(f"   {len(df)} étudiants trouvés")
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier {filepath} n'existe pas")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        sys.exit(1)

    # 2. Extraire y (labels - les maisons)
    if 'Hogwarts House' not in df.columns:
        print("❌ Erreur : Colonne 'Hogwarts House' introuvable")
        sys.exit(1)

    y = df['Hogwarts House'].copy()

    # Vérifier qu'il n'y a pas de NaN dans y
    nan_count = y.isna().sum()
    if nan_count > 0:
        print(f"⚠️  Attention : {nan_count} valeurs manquantes dans "
              f"'Hogwarts House'")
        # On retire les lignes avec NaN dans y
        valid_indices = y.notna()
        y = y[valid_indices]
        df = df[valid_indices]
        print(f"   → {len(y)} étudiants conservés après nettoyage")

    # 3. Extraire X (features - les cours)
    # Colonnes à exclure (non-numériques ou non-pertinentes)
    non_feature_columns = ['Index', 'Hogwarts House', 'First Name',
                           'Last Name', 'Birthday', 'Best Hand']

    # Déterminer les features à utiliser
    if selected_features is not None:
        # Utiliser les features sélectionnées (d'après l'analyse EDA)
        feature_columns = selected_features
        print("🎯 Utilisation des features sélectionnées (analyse EDA)")
    else:
        # Utiliser toutes les colonnes numériques
        feature_columns = []
        for col in df.columns:
            if col not in non_feature_columns:
                # Vérifier que c'est bien numérique
                if df[col].dtype in ['float64', 'int64']:
                    feature_columns.append(col)
        print("📊 Utilisation de toutes les features numériques")

    if len(feature_columns) == 0:
        print("❌ Erreur : Aucune feature numérique trouvée")
        sys.exit(1)

    # Vérifier que toutes les features demandées existent
    for col in feature_columns:
        if col not in df.columns:
            print(f"❌ Erreur : Feature '{col}' introuvable dans le dataset")
            sys.exit(1)

    # Convertir X en numpy array pour les calculs matriciels
    X = df[feature_columns].values

    print(f"✅ Features sélectionnées : {len(feature_columns)} cours")
    for i, feat in enumerate(feature_columns, 1):
        print(f"   {i}. {feat}")
    print(f"\n✅ Labels : {y.nunique()} maisons")
    print(f"   {sorted(y.unique())}")
    print(f"\n📐 Shape de X : {X.shape} (samples, features)")

    return X, y, feature_columns


def handle_missing_values(X):
    """Remplace les valeurs manquantes (NaN) par la moyenne de leur colonne"""
    # Créer une copie pour ne pas modifier l'original
    X_clean = X.copy()

    m, n = X.shape  # m = nombre d'étudiants, n = nombre de features
    total_nans = 0

    # Pour chaque colonne (chaque feature)
    for j in range(n):
        # Extraire la colonne j
        column = X[:, j]

        # Calculer la moyenne
        total = 0.0
        count = 0

        # Parcourir toutes les valeurs de la colonne
        for i in range(m):
            val = column[i]
            # Vérifier si c'est un NaN
            if not np.isnan(val):
                total += val
                count += 1

        # Si toute la colonne est NaN
        if count == 0:
            mean = 0.0
            print(f"⚠️  Colonne {j} entièrement NaN → moyenne = 0")
        else:
            mean = total / count

        # Compter et remplacer les NaN dans cette colonne
        # Utiliser un masque booléen pour identifier les NaN
        nan_mask = np.isnan(X_clean[:, j])
        nan_count = np.sum(nan_mask)

        if nan_count > 0:
            # Remplacer tous les NaN de cette colonne par la moyenne
            X_clean[nan_mask, j] = mean
            total_nans += nan_count

    print(f"✅ Valeurs manquantes traitées : {total_nans} NaN remplacés")
    if total_nans > 0:
        print("   Stratégie : remplacement par la moyenne de chaque colonne")

    return X_clean


def standardize(X):
    """Standardise les données avec le z-score (normalisation)
    Description:
        Pour chaque colonne :
        1. Calculer manuellement la moyenne μ
        2. Calculer manuellement l'écart-type σ
           σ = √(Σ(x - μ)² / m)
        3. Normaliser : x_norm = (x - μ) / σ
    """
    m, n = X.shape
    X_norm = X.copy()

    # Tableaux pour stocker les moyennes et écarts-types
    means = np.zeros(n)
    stds = np.zeros(n)

    # Pour chaque colonne (chaque feature)
    for j in range(n):
        column = X[:, j]

        # 1. Calculer la moyenne
        total = 0.0
        for i in range(m):
            total += column[i]
        mean = total / m
        means[j] = mean

        # 2. Calculer l'écart-type
        # std = √(Σ(x - mean)² / m)
        sum_squared_diff = 0.0
        for i in range(m):
            diff = column[i] - mean
            sum_squared_diff += diff * diff

        variance = sum_squared_diff / m
        std = variance ** 0.5  # Racine carrée
        stds[j] = std

        # 3. Normaliser la colonne
        # Si std = 0 (colonne constante), on ne divise pas pour éviter NaN
        if std > 0:
            X_norm[:, j] = (column - mean) / std
        else:
            # Colonne constante → on centre juste (- mean)
            X_norm[:, j] = column - mean
            print(f"⚠️  Colonne {j} a un écart-type nul → centrée seulement")

    print("✅ Standardisation terminée (z-score)")
    print(f"   Moyennes : min={means.min():.2f}, max={means.max():.2f}")
    print(f"   Écarts-types : min={stds.min():.2f}, max={stds.max():.2f}")
    print(f"   X_norm : min={X_norm.min():.2f}, max={X_norm.max():.2f}")

    return X_norm, means, stds


def add_intercept(X):
    """
    Ajoute une colonne de 1 au début de la matrice X (intercept/biais)

    Description:
        Ajoute une colonne de 1 tout à gauche de X.
        Cette colonne permet de calculer le terme d'intercept (θ₀)
        automatiquement dans le produit matriciel z = X @ θ

        Exemple:
            X = [[x1, x2],     →    X_intercept = [[1, x1, x2],
                 [x3, x4]]                          [1, x3, x4]]

        Shape : (m, n) → (m, n+1)
    """
    m = X.shape[0]  # Nombre d'échantillons

    # Créer une colonne de 1 de taille (m,)
    ones = np.ones(m)

    # Concaténer horizontalement : [colonne de 1] + [X]
    # np.c_ permet de concaténer des colonnes
    X_with_intercept = np.c_[ones, X]

    print("✅ Colonne d'intercept ajoutée")
    print(f"   Shape avant : {X.shape}")
    print(f"   Shape après : {X_with_intercept.shape}")

    return X_with_intercept


def main():
    """Point d'entrée principal du programme"""
    # Récupérer le fichier
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = 'data/dataset_train.csv'

    print("=" * 60)
    print("⚡ POUDLARD - ENTRAÎNEMENT DU MODÈLE ⚡")
    print("=" * 60)
    print()

    # Étape 1 : Charger les données
    print("📂 ÉTAPE 1/4 : Chargement des données")
    print("-" * 60)

    # Option 1 : Utiliser toutes les features (par défaut)
    X, y, feature_names = load_data(filepath)

    # Option 2 : Utiliser seulement les meilleures features (d'après pair_plot)
    # D'après ton analyse, tu pourrais sélectionner par exemple :
    # selected = ['Herbology', 'Ancient Runes', 'Astronomy',
    #             'Defense Against the Dark Arts']
    # X, y, feature_names = load_data(filepath, selected_features=selected)

    print()

    # Étape 2 : Gérer les NaN
    print("🔧 ÉTAPE 2/4 : Gestion des valeurs manquantes")
    print("-" * 60)
    X = handle_missing_values(X)
    # Vérification : plus aucun NaN
    remaining_nans = np.sum(np.isnan(X))
    if remaining_nans == 0:
        print("✅ Vérification : 0 NaN restant dans X")
    else:
        print(f"⚠️  Attention : {remaining_nans} NaN encore présents!")
    print()

    # Étape 3 : Standardiser (normalisation z-score)
    print("📊 ÉTAPE 3/4 : Standardisation (z-score)")
    print("-" * 60)
    X, means, stds = standardize(X)
    print()

    # Étape 4 : Ajouter la colonne d'intercept
    print("➕ ÉTAPE 4/4 : Ajout de la colonne d'intercept")
    print("-" * 60)
    X = add_intercept(X)
    print()

    print("=" * 60)
    print("✅ PRÉPARATION DES DONNÉES TERMINÉE")
    print("=" * 60)
    print(f"   Shape finale de X : {X.shape}")
    print(f"   Nombre d'échantillons : {X.shape[0]}")
    print(f"   Nombre de features (+intercept) : {X.shape[1]}")
    print()

    # TODO : Étape 5 - Entraîner le modèle
    # TODO : Étape 6 - Sauvegarder les poids


if __name__ == '__main__':
    main()
