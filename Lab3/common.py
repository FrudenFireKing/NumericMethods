"""Общие вспомогательные функции для лабораторной работы №3"""


#  чтение входных файлов

def read_lines(path):
    # Непустые строки файла без комментариев (всё после #)
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.split("#")[0].strip()
            if s:
                out.append(s)
    return out


def parse_floats(s):
    return [float(t) for t in s.replace(",", " ").split()]


def read_task1(path):
    # Формат 3.1: 'узлы xi' | x*   (значения yi вычисляются как ln(xi))
    t = [parse_floats(line) for line in read_lines(path)]
    return t[0], t[1][0]


def read_task2(path):
    # Формат 3.2: 'узлы xi' | 'значения yi' | x*
    t = [parse_floats(line) for line in read_lines(path)]
    if len(t[0]) != len(t[1]):
        raise ValueError("число узлов не совпадает с числом значений")
    return t[0], t[1], t[2][0]


def read_task3(path):
    # Формат 3.3: 'узлы xi' | 'значения yi'
    t = [parse_floats(line) for line in read_lines(path)]
    if len(t[0]) != len(t[1]):
        raise ValueError("число узлов не совпадает с числом значений")
    return t[0], t[1]


def read_task4(path):
    # Формат 3.4: 'узлы xi' | 'значения yi' | x*
    return read_task2(path)


def read_task5(path):
    # Формат 3.5: 'a b' | 'h1 h2'
    t = [parse_floats(line) for line in read_lines(path)]
    return t[0][0], t[0][1], t[1][0], t[1][1]


#  многочлены (список коэффициентов от младшей степени к старшей)

def poly_add(p, q):
    n = max(len(p), len(q))
    r = [0.0] * n
    for i in range(n):
        r[i] = (p[i] if i < len(p) else 0.0) + (q[i] if i < len(q) else 0.0)
    return r


def poly_mul(p, q):
    r = [0.0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            r[i + j] += a * b
    return r


def poly_scale(p, c):
    return [a * c for a in p]


def poly_from_root(x0):
    # многочлен (x - x0):  [-x0, 1]
    return [-x0, 1.0]


def poly_eval(p, x):
    # Схема Горнера
    s = 0.0
    for a in reversed(p):
        s = s * x + a
    return s


def fmt_poly(p, var="x", prec=6):
    # Строка вида 'c3*x^3 + c2*x^2 + c1*x + c0'
    terms = []
    for i in range(len(p) - 1, -1, -1):
        c = p[i]
        if abs(c) < 1e-12:
            continue
        sign = "-" if c < 0 else "+"
        a = abs(c)
        body = f"{a:.{prec}f}"
        if i == 0:
            mon = ""
        elif i == 1:
            mon = f"*{var}"
        else:
            mon = f"*{var}^{i}"
        terms.append(f"{sign} {body}{mon}")
    if not terms:
        return "0"
    if terms[0].startswith("+ "):
        terms[0] = terms[0][2:]
    else:
        terms[0] = "-" + terms[0][1:].strip()
    return " ".join(terms)


#  решение СЛАУ (метод Гаусса с выбором главного элемента)

def solve_linear(A, b):
    # Используется для нормальных систем МНК
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[p][k]) < 1e-14:
            raise ValueError("Вырожденная система — решение не существует")
        M[k], M[p] = M[p], M[k]
        for i in range(k + 1, n):
            m = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= m * M[k][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i]
    return x


#  оформление графиков

def style_axes(ax):
    # Единый стиль: без верхней/правой рамок, пунктирная сетка
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")