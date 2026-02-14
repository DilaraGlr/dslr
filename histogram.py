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
    """Affiche les histogrammes thème Harry Potter avec les couleurs des maisons"""
    # 1. Couleurs des 4 maisons de Poudlard
    house_colors = {
        'Gryffindor': '#AE0001',    # Rouge
        'Slytherin': '#2A623D',     # Vert vif
        'Ravenclaw': '#0E1A40',     # Bleu
        'Hufflepuff': '#ECB939'     # Jaune
    }

    # 2. Récupérer les 4 maisons (sans les NaN)
    houses = df['Hogwarts House'].dropna().unique()

    # 3. Créer une grille de subplots
    # Ex: 13 cours → grille 5x3
    n_courses = len(courses)
    n_cols = 3
    # Division arrondie vers le haut
    n_rows = (n_courses + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))

    # Style parchemin façon grimoire de Poudlard
    fig.patch.set_facecolor('#F5E6D3')  # Beige clair extérieur

    # Titre principal stylisé
    fig.suptitle('⚡ Poudlard - Distribution des Notes par Cours ⚡',
                 fontsize=18, fontweight='bold', color='#D3A625', y=0.995)

    # Aplatir la grille d'axes pour faciliter l'itération
    axes_flat = axes.ravel()

    # 4. Pour chaque cours:
    for i, course in enumerate(courses):
        ax = axes_flat[i]

        # Fond parchemin
        ax.set_facecolor('#FFF8E7')

        # Pour chaque maison, afficher un histogramme
        for house in houses:
            # Filtrer les données pour cette maison
            house_data = df[df['Hogwarts House'] == house][course]
            house_data = house_data.dropna()

            # Afficher l'histogramme avec couleurs des maisons
            color = house_colors.get(house, '#888888')
            ax.hist(house_data, alpha=0.7, label=house, bins=20,
                    color=color, edgecolor='#3D2817', linewidth=0.5)

        # Style du sous-graphique
        ax.set_title(course, fontsize=11, fontweight='bold', color='#3D2817')
        ax.set_xlabel('Notes', fontsize=9, color='#3D2817')
        ax.set_ylabel('Nombre d\'élèves', fontsize=9, color='#3D2817')

        # Grille dorée subtile
        ax.grid(True, alpha=0.3, color='#C9A961', linestyle=':')

        # Couleur des axes et ticks
        ax.tick_params(colors='#3D2817', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#8B6914')
            spine.set_linewidth(1.5)

        # Légende avec style parchemin
        legend = ax.legend(fontsize=7, framealpha=0.9,
                          facecolor='#F5E6D3', edgecolor='#8B6914')
        plt.setp(legend.get_texts(), color='#3D2817')

    # Masquer les subplots vides (si 13 cours dans une grille 5x3 = 15 cases)
    for i in range(n_courses, len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout()

    # 5. Afficher
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
