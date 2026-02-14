import sys
import matplotlib.pyplot as plt
from utils.preprocessing import load_csv


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


def select_features_for_plot(df, courses, max_features=4):
    """Sélectionne les N meilleures features pour le pair plot"""
    # Récupérer les 4 maisons
    houses = df['Hogwarts House'].dropna().unique()

    # Dictionnaire pour stocker le score de chaque cours
    course_scores = {}

    # Pour chaque cours
    for course in courses:
        # Calculer la moyenne pour chaque maison
        house_means = []

        for house in houses:
            # Filtrer les données pour cette maison
            house_data = df[df['Hogwarts House'] == house][course]
            house_data = house_data.dropna()

            if len(house_data) > 0:
                # Calculer la moyenne pour cette maison (version optimisée)
                values = house_data.values
                mean = sum(values) / len(values)
                house_means.append(mean)

        # Calculer la variance entre les moyennes des maisons
        # Plus la variance est élevée, plus les maisons sont séparées
        if len(house_means) > 1:
            # Moyenne des moyennes
            overall_mean = sum(house_means) / len(house_means)

            # Variance = Σ(mean - overall_mean)² / n (version optimisée)
            variance = sum((mean - overall_mean) ** 2
                           for mean in house_means) / len(house_means)

            course_scores[course] = variance
        else:
            course_scores[course] = 0

    # Trier les cours par score décroissant
    sorted_courses = sorted(course_scores.items(),
                            key=lambda x: x[1], reverse=True)

    # Garder les N meilleurs cours
    best_courses = [course for course, _ in sorted_courses[:max_features]]

    # Afficher les cours sélectionnés
    print(f"\nCours sélectionnés pour le pair plot (top {max_features}):")
    for i, (course, score) in enumerate(sorted_courses[:max_features], 1):
        print(f"  {i}. {course} (score: {score:.2f})")

    return best_courses


def plot_pair_plot(df, features):
    """Affiche un pair plot (matrice de scatter plots)"""
    # 1. Récupérer le nombre de features
    n = len(features)  # Ex: 4 → grille 4×4

    # 2. Récupérer les maisons
    houses = df['Hogwarts House'].dropna().unique()

    # 3. Définir les couleurs des maisons (thème Harry Potter)
    house_colors = {
        'Gryffindor': '#AE0001',
        'Slytherin': '#2A623D',
        'Ravenclaw': '#0E1A40',
        'Hufflepuff': '#ECB939'
    }

    # 4. Créer la grille de subplots
    fig, axes = plt.subplots(n, n, figsize=(3.5 * n, 3.5 * n))

    fig.patch.set_facecolor('#F5E6D3')

    fig.suptitle('⚡ Poudlard - Pair Plot des Meilleurs Cours ⚡',
                 fontsize=18, fontweight='bold', color='#D3A625', y=0.995)

    # 5. Remplir chaque case de la grille
    for i in range(n):
        for j in range(n):
            ax = axes[i, j] if n > 1 else axes

            ax.set_facecolor('#FFF8E7')

            # Cas 1 : DIAGONALE (i == j) → Histogramme
            if i == j:
                feature = features[i]

                # Histogramme pour chaque maison
                for house in houses:
                    house_data = df[df['Hogwarts House'] == house][feature]
                    house_data = house_data.dropna()

                    if len(house_data) > 0:
                        color = house_colors.get(house, '#888888')
                        ax.hist(house_data.values, alpha=0.7, bins=15,
                                color=color, edgecolor='#3D2817',
                                linewidth=0.5)

                ax.set_title(feature, fontsize=10, fontweight='bold',
                             color='#3D2817')

            # Cas 2 : HORS DIAGONALE → Scatter plot
            else:
                feature_x = features[j]  # Feature en X (colonne)
                feature_y = features[i]  # Feature en Y (ligne)

                # Scatter plot pour chaque maison
                for house in houses:
                    house_data = df[df['Hogwarts House'] == house]
                    house_data = house_data[[feature_x, feature_y]].dropna()

                    if len(house_data) > 0:
                        color = house_colors.get(house, '#888888')
                        ax.scatter(house_data[feature_x].values,
                                   house_data[feature_y].values,
                                   c=color, alpha=0.6, s=15,
                                   edgecolors='#3D2817', linewidth=0.3)

            # 6. Style des axes
            # Labels sur les bords uniquement
            if i == n - 1:  # Dernière ligne
                ax.set_xlabel(features[j], fontsize=9, color='#3D2817',
                              fontweight='bold')
            else:
                ax.set_xlabel('')

            if j == 0:  # Première colonne
                ax.set_ylabel(features[i], fontsize=9, color='#3D2817',
                              fontweight='bold')
            else:
                ax.set_ylabel('')

            ax.grid(True, alpha=0.3, color='#C9A961', linestyle=':')

            ax.tick_params(colors='#3D2817', labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('#8B6914')
                spine.set_linewidth(1)

    # 7. Légende globale (en dehors de la grille)
    legend_elements = []
    for house in houses:
        color = house_colors.get(house, '#888888')
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color, markersize=8,
                       label=house, markeredgecolor='#3D2817',
                       markeredgewidth=0.5)
        )

    fig.legend(handles=legend_elements, loc='upper right',
               fontsize=11, framealpha=0.95, facecolor='#F5E6D3',
               edgecolor='#8B6914', bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    plt.show()


def main():
    # 1. Charger
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'data/dataset_train.csv'

    print(f"📚 Chargement de {filename}...")
    df = load_csv(filename)

    if df is None:
        sys.exit(1)

    # 2. Récupérer les cours
    courses = get_courses(df)

    # 3. Sélectionner les meilleurs
    selected = select_features_for_plot(df, courses, max_features=4)

    # 4. Afficher le pair plot
    print("\n🎨 Génération du pair plot magique...\n")
    plot_pair_plot(df, selected)

    # 5. Conclusion
    print("\n📋 Recommandation pour la logistic regression:")
    print("Utiliser les features qui séparent bien les maisons")
    print("(visibles dans le pair plot)")


if __name__ == '__main__':
    main()
