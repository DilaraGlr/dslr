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
    """Mettre toutes les features à la même échelle (z-score normalization)
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


def sigmoid(z):
    """
    Transforme un score brut en probabilité (entre 0 et 1)"""
    # Limiter z pour éviter l'overflow numérique
    z = np.clip(z, -500, 500)

    # Appliquer la formule sigmoid
    return 1 / (1 + np.exp(-z))


def cost_function(h, y):
    """
    Calcule la log-loss (Binary Cross-Entropy)

    Args:
        h: numpy array (m,) - probabilités prédites par sigmoid (entre 0 et 1)
        y: numpy array (m,) - vraies valeurs binaires (0 ou 1)

    Returns:
        float - valeur de la loss (plus elle est basse, mieux c'est)

    Formule:
        J = -1/m * Σ(y * log(h) + (1 - y) * log(1 - h))

        - y * log(h)       : pénalise si la réponse est 1 et h proche de 0
        - (1-y) * log(1-h) : pénalise si la réponse est 0 et h proche de 1
    """
    m = len(y)

    # Clipper h pour éviter log(0) qui donne -inf et crashe tout
    h = np.clip(h, 1e-15, 1 - 1e-15)

    # Calculer la log-loss
    # Partie 1 : cas où la vraie réponse est 1 → y * log(h)
    # Partie 2 : cas où la vraie réponse est 0 → (1 - y) * log(1 - h)
    loss = -1 / m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

    return loss


def compute_gradient(X, h, y):
    """
    Calcule le gradient de la loss par rapport aux poids theta

    Args:
        X: numpy array (m, n+1) - données avec intercept
        h: numpy array (m,)    - probabilités prédites par sigmoid
        y: numpy array (m,)    - vraies valeurs binaires (0 ou 1)

    Returns:
        grad: numpy array (n+1,) - gradient pour chaque poids

    Formule vectorisée:
        grad = (1/m) * X.T @ (h - y)

        - (h - y)   : erreur entre prédiction et vérité (m,)
        - X.T       : X transposé, shape (n+1, m)
        - X.T @ (h-y) : produit matriciel → un gradient par poids (n+1,)
        - 1/m       : moyenne sur tous les étudiants
    """
    m = len(y)

    grad = (1 / m) * X.T @ (h - y)

    return grad


def train_one_vs_all(X, y, learning_rate=0.1, epochs=1000):
    """
    Entraîne 4 classifieurs binaires (un par maison)

    Args:
        X: numpy array (m, n+1) - données avec intercept
        y: pandas Series        - labels (noms des maisons)
        learning_rate: float    - taille du pas du gradient descent
        epochs: int             - nombre d'itérations d'entraînement

    Returns:
        weights: dict - {'Gryffindor': theta, 'Slytherin': theta, ...}
                        un tableau de poids pour chaque maison

    Description:
        Pour chaque maison :
        1. Convertir y en vecteur binaire (1 = cette maison, 0 = autre)
        2. Initialiser theta à zéro
        3. Répéter `epochs` fois :
           a. h = sigmoid(X @ theta)        → probabilités
           b. loss = compute_loss(h, y_bin) → mesure l'erreur
           c. grad = compute_gradient(...)  → direction de correction
           d. theta = theta - lr * grad     → ajuster les poids
    """
    # Récupérer les 4 maisons uniques
    houses = sorted(y.unique())

    # Dictionnaire pour stocker les poids de chaque maison
    weights = {}

    print(f"Entraînement : {len(houses)} maisons, "
          f"{epochs} epochs, lr={learning_rate}")
    print()

    # Pour chaque maison
    for house in houses:
        print(f"  🏰 {house}...")

        # 1. Convertir y en vecteur binaire
        # 1 = c'est cette maison, 0 = c'est une autre maison
        y_binary = (y == house).astype(int).values

        # 2. Initialiser theta à zéro (un poids par feature + biais)
        theta = np.zeros(X.shape[1])

        # 3. Boucle de gradient descent
        for epoch in range(epochs):
            # a. Calculer les probabilités prédites
            h = sigmoid(X @ theta)

            # b. Mesurer l'erreur (loss)
            loss = cost_function(h, y_binary)

            # c. Calculer le gradient
            grad = compute_gradient(X, h, y_binary)

            # d. Ajuster les poids
            theta = theta - learning_rate * grad

            # Afficher la progression toutes les 200 epochs
            if (epoch + 1) % 200 == 0:
                print(f"     epoch {epoch + 1}/{epochs} → loss = {loss:.4f}")

        # Sauvegarder les poids de cette maison
        weights[house] = theta

    print()
    print(f"✅ Entraînement terminé ! {len(weights)} modèles entraînés")

    return weights


def save_weights(weights, means, stds, feature_names, filepath='weights.json'):
    """
    Sauvegarde les poids et paramètres de normalisation dans un fichier JSON

    Args:
        weights: dict        - {'Gryffindor': theta, ...} (numpy arrays)
        means: numpy array   - moyennes de chaque feature (de standardize)
        stds: numpy array    - écarts-types de chaque feature (de standardize)
        feature_names: list  - noms des features (de load_data)
        filepath: str        - chemin du fichier JSON à créer

    Description:
        Sauvegarde tout ce dont logreg_predict.py aura besoin :
        - feature_names : pour savoir quelles colonnes extraire du test set
        - means, stds   : pour normaliser le test set de la même façon
        - weights       : pour calculer les probabilités et prédire

    Note:
        JSON ne comprend pas les numpy arrays → .tolist() pour convertir
    """
    # Créer le dictionnaire avec toutes les données nécessaires
    data = {
        'feature_names': feature_names,
        'means': means.tolist(),
        'stds': stds.tolist(),
        'weights': {
            house: theta.tolist()
            for house, theta in weights.items()
        }
    }

    # Écrire dans le fichier JSON
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Poids sauvegardés dans : {filepath}")
    print(f"   Features : {len(feature_names)}")
    print(f"   Maisons  : {list(weights.keys())}")


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

    # Étape 5 : Entraîner le modèle
    print("🧠 ÉTAPE 5 : Entraînement (One vs All)")
    print("-" * 60)
    weights = train_one_vs_all(X, y, learning_rate=0.1, epochs=1000)
    print()

    # Étape 6 : Sauvegarder les poids
    print("💾 ÉTAPE 6 : Sauvegarde des poids")
    print("-" * 60)
    save_weights(weights, means, stds, feature_names)
    print()
    print("=" * 60)
    print("⚡ ENTRAÎNEMENT TERMINÉ - Prêt pour la prédiction !")
    print("=" * 60)


if __name__ == '__main__':
    main()
