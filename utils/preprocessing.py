import pandas as pd


def load_csv(filename):
    """Charge un fichier CSV"""
    try:
        data = pd.read_csv(filename)
        return data
    except FileNotFoundError:
        print(f"Erreur: Le fichier {filename} n'existe pas")
        return None
    except Exception as e:
        print(f"Erreur: {e}")
        return None


def handle_nan(data):
    """Gère les valeurs manquantes en les remplaçant par la moyenne"""
    if data is None:
        print("❌ Aucune donnée à traiter")
        return None

    print("🔧 Traitement des valeurs manquantes...")

    # Créer une copie pour ne pas modifier l'original
    data_cleaned = data.copy()

    total_nans = 0

    # Pour chaque colonne du DataFrame
    for col in data_cleaned.columns:
        # Ignorer la colonne Index
        if col == 'Index':
            continue

        # Vérifier manuellement si la colonne est numérique
        if data_cleaned[col].dtype not in ['float64', 'int64']:
            continue

        # Compter les NaN dans cette colonne
        nan_count = 0
        for val in data_cleaned[col]:
            if val != val:  # Détecter NaN (NaN != NaN)
                nan_count += 1

        if nan_count == 0:
            continue  # Pas de NaN, passer à la suivante

        # Calculer la moyenne MANUELLEMENT (sans .mean())
        total = 0
        count = 0
        for val in data_cleaned[col]:
            if val == val:  # Si ce n'est pas NaN
                total += val
                count += 1

        if count > 0:
            mean_value = total / count

            # Remplacer les NaN par la moyenne
            for idx in range(len(data_cleaned)):
                current_val = data_cleaned.iloc[idx][col]
                if current_val != current_val:  # Si c'est NaN
                    data_cleaned.at[idx, col] = mean_value

            total_nans += nan_count
            print(f"   • {col}: {nan_count} NaN → {mean_value:.2f}")

    print(f"✅ Total: {total_nans} valeurs traitées\n")
    return data_cleaned


def normalize(data):
    """Normalise les données avec z-score"""
    # 1. Créer une copie
    data_copy = data.copy()

    # 2. Pour chaque colonne numérique (sauf Index):
    for col in data_copy.columns:
        # Ignorer la colonne Index
        if col == 'Index':
            continue

        # Vérifier si la colonne est numérique
        if data_copy[col].dtype not in ['float64', 'int64']:
            continue

        values = data_copy[col].values

        # Calculer la moyenne μ
        # Formule: μ = (Σ valeurs) / nombre_valeurs
        sum_val = 0
        count = 0
        for val in values:
            # Ignorer les NaN (car NaN != NaN)
            if val == val:
                sum_val += val
                count += 1

        if count == 0:
            continue

        mean = sum_val / count

        # Calculer l'écart-type σ
        # Formule: σ = √(Σ(x - μ)² / n)
        sum_squared_diff = 0
        for val in values:
            if val == val:  # Ignorer les NaN
                # (x - μ)² : carré de la différence avec la moyenne
                diff = val - mean
                sum_squared_diff += diff * diff

        # Variance = moyenne des carrés des écarts
        variance = sum_squared_diff / count
        # Écart-type = racine carrée de la variance
        std = variance ** 0.5

        # Pour chaque valeur: Appliquer z = (x - μ) / σ
        if std != 0:  # Éviter la division par zéro
            for i in range(len(values)):
                if values[i] == values[i]:  # Si pas NaN
                    # Transformer la valeur en z-score
                    data_copy.iloc[i, data_copy.columns.get_loc(col)] = \
                        (values[i] - mean) / std
        else:
            print(f"Attention: '{col}' a un écart-type de zéro.")

    # 3. Retourner le DataFrame normalisé
    return data_copy


def select_features(data, feature_names):
    """Sélectionne les colonnes utiles du DataFrame"""

    # 1. Vérifications
    if data is None:
        print("❌ Aucune donnée fournie")
        return None

    if not feature_names or len(feature_names) == 0:
        print("❌ Aucune feature spécifiée")
        return None

    # 2. Vérifier que les colonnes existent (optionnel)
    for feature in feature_names:
        if feature not in data.columns:
            print(f"❌ La feature '{feature}' n'existe pas dans les données")
            return None

    # 3. Sélectionner les colonnes
    selected_data = data[feature_names]

    print(f"✅ {len(feature_names)} features sélectionnées")
    return selected_data
