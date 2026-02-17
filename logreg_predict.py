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

    # Laisser weights en listes pour l'instant (converti plus tard dans predict)
    weights = data['weights']

    print(f"✅ Poids chargés depuis : {filepath}")
    print(f"   Features : {len(feature_names)}")
    print(f"   Maisons  : {list(weights.keys())}")

    return feature_names, means, stds, weights


def main():
    """Point d'entrée principal"""
    # Récupérer les fichiers
    filepath_test = sys.argv[1] if len(sys.argv) > 1 else 'data/dataset_test.csv'
    filepath_weights = sys.argv[2] if len(sys.argv) > 2 else 'weights.json'

    print("=" * 60)
    print("⚡ POUDLARD - PRÉDICTION DES MAISONS ⚡")
    print("=" * 60)
    print()

    # Étape 1 : Charger les poids
    print("📂 ÉTAPE 1 : Chargement des poids")
    print("-" * 60)
    feature_names, means, stds, weights = load_weights(filepath_weights)
    print()

    # TODO : Étape 2 - Charger et préparer le test set
    # TODO : Étape 3 - Prédire les maisons
    # TODO : Étape 4 - Sauvegarder les prédictions


if __name__ == '__main__':
    main()
