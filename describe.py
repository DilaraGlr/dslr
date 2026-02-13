import sys
import pandas as pd
from utils import stats


def main():
    if len(sys.argv) != 2:
        print("Usage: python describe.py <dataset.csv>")
        sys.exit(1)

    try:
        df = pd.read_csv(sys.argv[1])
    except Exception as e:
        print(f"Erreur: {e}")
        sys.exit(1)

    # Filtrage des colonnes numériques uniquement
    numeric_cols = [col for col in df.columns if df[col].dtype in [
        'float64', 'int64'] and col.lower() != "index"]

    stat_titles = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]

    NAME_W = 20
    VAL_W = 14

    # 1. Affichage de l'en-tête
    header = f"{'Feature':<{NAME_W}}"
    for title in stat_titles:
        header += f"{title:>{VAL_W}}"
    print(header)
    print("-" * (NAME_W + len(stat_titles) * VAL_W))

    # 2. Calcul et affichage par matière
    for col in numeric_cols:
        data = df[col].values

        c = stats.count(data)
        if c == 0:
            continue

        m = stats.mean(data, c)
        s = stats.std(data, c, m)
        mini = stats.min(data)
        maxi = stats.max(data)

        sorted_data = stats.sortData(data)
        q25 = stats.get_quartile(sorted_data, c, 0.25)
        q50 = stats.get_quartile(sorted_data, c, 0.50)
        q75 = stats.get_quartile(sorted_data, c, 0.75)

        row = f"{col[:NAME_W-1]:<{NAME_W}}"
        row += f"{c:>{VAL_W}.6f}"
        row += f"{m:>{VAL_W}.6f}"
        row += f"{s:>{VAL_W}.6f}"
        row += f"{mini:>{VAL_W}.6f}"
        row += f"{q25:>{VAL_W}.6f}"
        row += f"{q50:>{VAL_W}.6f}"
        row += f"{q75:>{VAL_W}.6f}"
        row += f"{maxi:>{VAL_W}.6f}"
        print(row)


if __name__ == "__main__":
    main()
