"""Общие вспомогательные функции для лабораторной работы №2"""


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
    # Формат: 'xmin xmax' | eps | x0_МПИ | x0_Ньютон
    t = [parse_floats(line) for line in read_lines(path)]
    xmin, xmax = t[0]
    eps = t[1][0]
    return xmin, xmax, eps, t[2][0], t[3][0]


def read_task2(path):
    # Формат: a | 'x1min x1max' | 'x2min x2max' | eps | x0_МПИ | x0_Ньютон
    t = [parse_floats(line) for line in read_lines(path)]
    a = t[0][0]
    x1r = (t[1][0], t[1][1])
    x2r = (t[2][0], t[2][1])
    eps = t[3][0]
    return a, x1r, x2r, eps, t[4], t[5]


#  векторы и маленькая СЛАУ 

def vec_sub(u, v):
    return [a - b for a, b in zip(u, v)]


def vec_inf(u):
    # ||u||_inf
    return max(abs(t) for t in u)


def solve_linear(A, b):
    # Гаусс с выбором главного элемента (для J*Delta = -F в методе Ньютона)
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(M[i][k]))
        if abs(M[p][k]) < 1e-14:
            raise ValueError("Вырожденная матрица Якоби — метод Ньютона неприменим")
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