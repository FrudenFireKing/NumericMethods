# Общие вспомогательные функции для лабораторной работы №1

import math


# Чтение входных файлов

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
    # '1 -2.5, 3e-1' -> [1.0, -2.5, 0.3]
    return [float(t) for t in s.replace(",", " ").split()]


def read_system(path):
    # Формат: n | n строк матрицы A | строка с вектором b
    lines = read_lines(path)
    n = int(parse_floats(lines[0])[0])
    A = [parse_floats(lines[i + 1]) for i in range(n)]
    b = parse_floats(lines[n + 1])
    if any(len(row) != n for row in A) or len(b) != n:
        raise ValueError("Размерности входных данных не согласованы")
    return n, A, b


def read_matrix_eps(path):
    # Формат: n | n строк матрицы | eps
    lines = read_lines(path)
    n = int(parse_floats(lines[0])[0])
    A = [parse_floats(lines[i + 1]) for i in range(n)]
    eps = parse_floats(lines[n + 1])[0]
    if any(len(row) != n for row in A):
        raise ValueError("Матрица должна быть квадратной")
    return n, A, eps


# линейная алгебра

def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def mat_mul(A, B):
    # Матричное произведение A*B
    n, m, p = len(A), len(B), len(B[0])
    C = [[0.0] * p for _ in range(n)]
    for i in range(n):
        for k in range(m):
            aik = A[i][k]
            if aik != 0.0:
                for j in range(p):
                    C[i][j] += aik * B[k][j]
    return C


def mat_vec(A, x):
    # A*x
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def transpose(A):
    return [list(col) for col in zip(*A)]


def mat_max_abs_diff(A, B):
    # ||A − B||_max — для проверок
    return max(abs(a - b) for ra, rb in zip(A, B) for a, b in zip(ra, rb))


def vec_max_abs_diff(u, v):
    return max(abs(a - b) for a, b in zip(u, v))


def vec_norm2(v):
    return math.sqrt(sum(x * x for x in v))


# форматированный вывод

def fmt(x, prec=6):
    if isinstance(x, complex):
        return f"{x.real:+.{prec}f}{x.imag:+.{prec}f}i"
    return f"{x:+.{prec}f}"


def print_matrix(A, title="", prec=6):
    if title:
        print(title)
    for row in A:
        print("   ".join(fmt(v, prec) for v in row))
    print()


def print_vector(v, title="", prec=6):
    if title:
        print(title)
    print("   ".join(fmt(x, prec) for x in v))
    print()