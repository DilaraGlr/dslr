def count(data: list) -> float:
    count = 0.0

    for el in data:
        if el == el:
            count += 1

    return count


def mean(data: list, n_el: float) -> float:
    total = 0

    for el in data:
        if el == el:
            total += el

    return total / n_el


def variance(data: list, n_count: float, mean: float) -> float:
    """Calcule la variance d'une liste de valeurs."""
    if n_count <= 1:
        return 0.0

    somme_diff_carres = 0.0
    for value in data:
        if value == value:
            diff = value - mean
            somme_diff_carres += diff * diff

    return somme_diff_carres / n_count


def skewness(data: list, n_count: float, mean: float, std: float) -> float:
    """
    Calcule l'asymétrie de la distribution.

    Args:
        data: valeurs numériques
        n_count: nombre de valeurs valides
        mean: moyenne déjà calculée
        std: écart-type déjà calculé

    Returns:
        float: le coefficient de skewness
            - Proche de 0 = distribution symétrique
            - Positif = queue à droite (valeurs extrêmes élevées)
            - Négatif = queue à gauche (valeurs extrêmes basses)

    Formule:
        skewness = (1/n) * Σ((x - mean)³) / std³
    """
    if n_count <= 1 or std == 0:
        return 0.0

    somme_diff_cubes = 0.0
    for value in data:
        if value == value:  # Ignorer les NaN
            diff = value - mean
            somme_diff_cubes += diff ** 3  # Au cube !

    # (1/n) * Σ(diff³) / std³
    return somme_diff_cubes / (n_count * (std ** 3))


def kurtosis(data: list, n_count: float, mean: float, std: float) -> float:
    """
    Calcule l'aplatissement de la distribution (excess kurtosis).

    Args:
        data: valeurs numériques
        n_count: nombre de valeurs valides
        mean: moyenne déjà calculée
        std: écart-type déjà calculé

    Returns:
        float: le coefficient de kurtosis
            - Proche de 0 = distribution normale (mesokurtic)
            - Positif = pics pointus, queues lourdes (leptokurtic)
            - Négatif = pics plats, queues légères (platykurtic)

    Formule:
        kurtosis = (1/n) * Σ((x - mean)⁴) / std⁴ - 3
    """
    if n_count <= 1 or std == 0:
        return 0.0

    somme_diff_quatre = 0.0
    for value in data:
        if value == value:  # Ignorer les NaN
            diff = value - mean
            somme_diff_quatre += diff ** 4  # Puissance 4 !

    # (1/n) * Σ(diff⁴) / std⁴ - 3
    return somme_diff_quatre / (n_count * (std ** 4)) - 3


def std(data: list, n_count: float, mean: float) -> float:
    if n_count <= 1:
        return 0.0

    somme_diff_carres = 0.0
    for value in data:
        if value == value:
            diff = value - mean
            somme_diff_carres += diff * diff

    variance = somme_diff_carres / n_count
    return variance ** 0.5


def min(data: list) -> float:
    mini = None
    for val in data:
        if val == val:
            mini = val
            break

    if mini is None:
        return 0.0

    for value in data:
        if value == value:
            if value < mini:
                mini = value

    return float(mini)


def max(data: list) -> float:
    maxi = None
    for val in data:
        if val == val:
            maxi = val
            break

    if maxi is None:
        return 0.0

    for value in data:
        if value == value:
            if value > maxi:
                maxi = value

    return float(maxi)


def sortData(data: list):
    clean_data = []
    for x in data:
        if x == x:
            clean_data.append(x)

    n = 0
    for _ in clean_data:
        n += 1

    for i in range(n):
        for j in range(0, n - i - 1):
            if clean_data[j] > clean_data[j + 1]:
                clean_data[j], clean_data[j +
                                          1] = clean_data[j + 1], clean_data[j]
    return clean_data


def get_quartile(data_sorted, n_count, percentile):
    if n_count == 0:
        return 0.0

    pos = percentile * (n_count - 1)

    # On sépare la partie entière et la partie décimale de l'index
    index_bas = int(pos)
    index_haut = index_bas + 1
    reste = pos - index_bas

    # Si l'index haut dépasse la liste, on renvoie le dernier élément
    if index_haut >= n_count:
        return float(data_sorted[index_bas])

    # Interpolation linéaire : ValeurBas + Reste * (ValeurHaut - ValeurBas)
    valeur = data_sorted[index_bas] + reste * \
        (data_sorted[index_haut] - data_sorted[index_bas])

    return float(valeur)
