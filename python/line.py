def slopes(x, y):
    rise = y[1:] - y[:-1]
    run = x[1:] - x[:-1]

    return rise / run


def intercepts(x, y):
    return y[1:] - slopes(x, y) * x[1:]


def crossings(x, y, q):
    crossings = (q - intercepts(x, y)) / slopes(x, y)

    return crossings[(crossings > x[:-1]) & (crossings < x[1:])]


def interval(x, y, q, p, precision=2):
    points = crossings(x, y, q)
    for low, high in [points[i:i + 2] for i in range(0, len(points), 2)]:
        if p > low and p < high:
            return (round(low, precision) + 0, round(high, precision) + 0)  # turn -0.00 -> 0.00
