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


def correlation(df, course1, course2):
    """Calcule la corrélation de Pearson entre 2 cours"""
    # 1. Récupérer les données des 2 cours
    df_clean = df[[course1, course2]].dropna()

    # Si pas assez de données
    if len(df_clean) < 2:
        return 0.0

    x_values = df_clean[course1].values
    y_values = df_clean[course2].values
    n = len(x_values)

    # 2. Calculer les moyennes μx et μy
    sum_x = 0
    sum_y = 0
    for i in range(n):
        sum_x += x_values[i]
        sum_y += y_values[i]

    mean_x = sum_x / n
    mean_y = sum_y / n

    # 3. Calculer les écarts-types σx et σy
    sum_sq_diff_x = 0
    sum_sq_diff_y = 0
    for i in range(n):
        diff_x = x_values[i] - mean_x
        diff_y = y_values[i] - mean_y
        sum_sq_diff_x += diff_x * diff_x
        sum_sq_diff_y += diff_y * diff_y

    std_x = (sum_sq_diff_x / n) ** 0.5
    std_y = (sum_sq_diff_y / n) ** 0.5

    # Si écart-type nul, pas de corrélation calculable
    if std_x == 0 or std_y == 0:
        return 0.0

    # 4. Calculer r avec la formule de Pearson
    sum_products = 0
    for i in range(n):
        sum_products += (x_values[i] - mean_x) * (y_values[i] - mean_y)

    r = sum_products / (n * std_x * std_y)

    return r


def find_most_similar_features(df, courses):
    """Trouve les 2 cours les plus corrélés"""
    max_corr = -1  # Initialiser à -1 (minimum possible)
    best_pair = None

    # 1. Pour toutes les paires de cours
    for i in range(len(courses)):
        for j in range(i + 1, len(courses)):  # i+1 pour éviter les doublons
            course1 = courses[i]
            course2 = courses[j]

            # Calculer leur corrélation
            r = correlation(df, course1, course2)

            # 2. Garder la paire avec la corrélation maximale
            # (en valeur absolue)
            # abs() car -0.99 est aussi fort que +0.99
            if abs(r) > max_corr:
                max_corr = abs(r)
                best_pair = (course1, course2, r)

    # 3. Retourner (cours1, cours2, r)
    if best_pair is None:
        print("⚠️ Aucune corrélation trouvée")
        return None, None, 0.0

    return best_pair


def plot_scatter(df, course1, course2):
    """Affiche un scatter plot thème Harry Potter avec les couleurs des maisons"""
    # 1. Récupérer les données avec la maison (sans NaN dans les cours)
    df_clean = df[[course1, course2, 'Hogwarts House']].dropna(
        subset=[course1, course2])

    if len(df_clean) == 0:
        print("Aucune donnée à afficher")
        return

    # Calculer la corrélation pour l'afficher
    r = correlation(df, course1, course2)

    # Interpréter la force de la corrélation
    if abs(r) > 0.7:
        strength = "Très forte"
    elif abs(r) > 0.5:
        strength = "Forte"
    elif abs(r) > 0.3:
        strength = "Modérée"
    else:
        strength = "Faible"

    # 2. Couleurs des 4 maisons de Poudlard
    house_colors = {
        'Gryffindor': '#AE0001',    # Rouge
        'Slytherin': '#2A623D',     # Vert plus vif
        'Ravenclaw': '#0E1A40',     # Bleu
        'Hufflepuff': '#ECB939'     # Jaune
    }

    # 3. Créer le scatter plot avec style Harry Potter
    fig, ax = plt.subplots(figsize=(12, 8))

    # Fond parchemin façon "grimoire de Poudlard"
    fig.patch.set_facecolor('#F5E6D3')  # Beige clair extérieur
    ax.set_facecolor('#FFF8E7')         # Parchemin clair pour le graphique

    # 4. Tracer les points par maison
    for house in house_colors.keys():
        house_data = df_clean[df_clean['Hogwarts House'] == house]
        if len(house_data) > 0:
            ax.scatter(house_data[course1], house_data[course2],
                       c=house_colors[house],
                       label=house,
                       alpha=0.8,
                       s=60,
                       edgecolors='#3D2817',  # Marron foncé
                       linewidth=0.8)

    # 5. Ligne de tendance dorée (couleur magique)
    x_values = df_clean[course1].values
    y_values = df_clean[course2].values
    n = len(x_values)
    mean_x = sum(x_values) / n
    mean_y = sum(y_values) / n

    numerator = 0
    denominator = 0
    for i in range(n):
        numerator += (x_values[i] - mean_x) * (y_values[i] - mean_y)
        denominator += (x_values[i] - mean_x) ** 2

    slope = None
    if denominator != 0:
        slope = numerator / denominator
        intercept = mean_y - slope * mean_x

        x_line = [min(x_values), max(x_values)]
        y_line = [slope * x + intercept for x in x_line]
        ax.plot(x_line, y_line, color='#6B4423', linewidth=2.5,
                linestyle='--', label='Tendance magique', alpha=0.9)

    # 6. Style du graphique
    # Grille dorée subtile
    ax.grid(True, alpha=0.3, color='#C9A961', linestyle=':')

    # Titre stylisé
    title = f'⚡ Poudlard - Analyse des Cours ⚡\n'
    title += f'{course1}  ×  {course2}\n'
    title += f'Corrélation: r = {r:.4f} ({strength})'
    ax.set_title(title, fontsize=16, fontweight='bold',
                 color='#D3A625', pad=20)

    # Labels avec style
    ax.set_xlabel(course1, fontsize=13, color='#3D2817', fontweight='bold')
    ax.set_ylabel(course2, fontsize=13, color='#3D2817', fontweight='bold')

    # Couleur des axes et ticks
    ax.tick_params(colors='#3D2817', labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#8B6914')  # Or foncé
        spine.set_linewidth(2)

    # Légende avec style parchemin
    legend = ax.legend(loc='best', fontsize=11, framealpha=0.95,
                       facecolor='#F5E6D3', edgecolor='#8B6914',
                       shadow=True)
    plt.setp(legend.get_texts(), color='#3D2817')

    # 7. Afficher
    plt.tight_layout()
    plt.show()


def main():
    """
    Fonction principale du programme

    Description:
        1. Charge le fichier CSV
        2. Trouve les 2 cours les plus corrélés
        3. Affiche le scatter plot avec la ligne de tendance
    """
    # 1. Récupère le nom du fichier
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'data/dataset_train.csv'

    # 2. Charge le fichier CSV
    print(f"Chargement du fichier: {filename}")
    df = load_csv(filename)

    if df is None:
        print("Erreur lors du chargement des données")
        sys.exit(1)

    # 3. Récupère la liste des cours
    courses = get_courses(df)
    print(f"{len(courses)} cours trouvés\n")

    # 4. Trouve les 2 cours les plus similaires
    print("Calcul des corrélations entre tous les cours...")
    result = find_most_similar_features(df, courses)

    if result is None or result[0] is None:
        print("Impossible de trouver des features similaires")
        sys.exit(1)

    course1, course2, r = result

    # 5. Affiche le résultat
    print("\nRésultat:")
    print("Les 2 features les plus similaires sont:")
    print(f"  • {course1}")
    print(f"  • {course2}")
    print(f"  → Corrélation: r = {r:.4f}\n")

    # 6. Affiche le scatter plot
    print("Affichage du scatter plot...")
    plot_scatter(df, course1, course2)


if __name__ == '__main__':
    main()
