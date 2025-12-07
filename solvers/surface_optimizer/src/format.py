size_units = [(1, 'B'), (2 ** 10, 'KB'), (2 ** 20, 'MB'), (2 ** 30, 'GB')]
time_units = [(1, 'ns'), (1e3, 'us'), (1e6, 'ms'), (1e9, 's'),
              (60 * 1e9, 'min'), (60 * 60 * 1e9, 'hr')]


def best_unit(x, units):
    i = 0
    while i + 1 < len(units) and x >= units[i + 1][0]:
        i += 1
    return i


def fmt(x, units):
    i = best_unit(x, units)
    return f'{x / units[i][0]:.2f} {units[i][1]}'


def sizefmt(x):
    return fmt(x, size_units)


def tfmt(t):
    return fmt(t * 1e9, time_units)
