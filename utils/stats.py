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
