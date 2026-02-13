import sys
import matplotlib.pyplot as plt
from utils.preprocessing import load_csv, handle_nan


def get_courses(df):
    """Récupère les noms des cours (colonnes numériques)"""
    # Colonnes à exclure (non-numériques)
    non_course_columns = ['Index', 'Hogwarts House', 'First Name',
                          'Last Name', 'Birthday', 'Best Hand']

    # Récupérer toutes les colonnes
    all_columns = df.columns.tolist()

    # Filtrer pour garder uniquement les cours
    courses = []
    for col in all_columns:
        if col not in non_course_columns:
            courses.append(col)

    return courses


def plot_histograms(df, courses):
    """Affiche les histogrammes de tous les cours"""
    # 1. Récupérer les 4 maisons (sans les NaN)
    houses = df['Hogwarts House'].dropna().unique()

    # 2. Créer une grille de subplots
    # Ex: 13 cours → grille 5x3
    n_courses = len(courses)
    n_cols = 3
    # Division arrondie vers le haut
    n_rows = (n_courses + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
    fig.suptitle('Distribution des notes par cours et par maison', fontsize=16)

    # Aplatir la grille d'axes pour faciliter l'itération
    axes_flat = axes.ravel()  # ou axes.flatten()

    # 3. Pour chaque cours:
    for i, course in enumerate(courses):
        ax = axes_flat[i]

        # Pour chaque maison, afficher un histogramme
        for house in houses:
            # Filtrer les données pour cette maison
            house_data = df[df['Hogwarts House'] == house][course]
            house_data = house_data.dropna()
            # Afficher l'histogramme (alpha=0.5 pour transparence)
            ax.hist(house_data, alpha=0.5, label=house, bins=20)

        # Ajouter le titre et les labels
        ax.set_title(course, fontsize=10)
        ax.set_xlabel('Notes')
        ax.set_ylabel('Nombre d\'élèves')
        ax.legend(fontsize=8)

    # Masquer les subplots vides (si 13 cours dans une grille 5x3 = 15 cases)
    for i in range(n_courses, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout()

    # 4. Afficher
    plt.show()


def main():
    """Point d'entrée principal du programme"""
    # Déterminer le fichier à charger
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'data/dataset_train.csv'

    # Charger les données
    print(f"Chargement du fichier: {filename}")
    df = load_csv(filename)

    if df is None:
        print("Erreur lors du chargement des données")
        sys.exit(1)

    # Gérer les valeurs manquantes
    df = handle_nan(df)

    # Récupérer les cours
    courses = get_courses(df)
    print(f"{len(courses)} cours trouvés\n")

    # Afficher les histogrammes
    print("Génération des histogrammes...")
    print("\nRegardez quel cours a des distributions similaires "
          "entre les 4 maisons")
    print("(hauteurs des barres équilibrées)\n")

    plot_histograms(df, courses)


if __name__ == '__main__':
    main()
