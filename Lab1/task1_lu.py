"""
Лабораторная работа 1.1 — LU-разложение с выбором главного элемента.

Строится разложение P*A = L*U (частичный выбор главного элемента по столбцу),
затем:
  * решается СЛАУ A*x = b  (Ly = Pb — прямая подстановка, Ux = y — обратная);
  * det A = (-1)^(число обменов строк) * произведение диагонали U;
  * A^(-1): решаем n систем A*x = e_i, решения — столбцы обратной матрицы;
  * проверки: L*U = P*A и A*A^(-1) = E

Вход: data/task1.txt (n; матрица; вектор b). Запуск: python task1_lu.py [файл]
"""

import sys

from common import (read_system, identity, mat_mul, mat_vec, transpose,
                    mat_max_abs_diff, print_matrix, print_vector)

PIVOT_EPS = 1e-12


def lu_decompose(A):
    n = len(A)
    U = [row[:] for row in A]
    L = identity(n)
    perm = list(range(n))
    swaps = 0

    for k in range(n):
        p = k
        for i in range(k + 1, n):
            if abs(U[i][k]) > abs(U[p][k]):
                p = i
        if abs(U[p][k]) < PIVOT_EPS:
            raise ValueError("Матрица вырождена: ведущий элемент ~ 0")
        if p != k:
            U[k], U[p] = U[p], U[k]
            perm[k], perm[p] = perm[p], perm[k]
            for j in range(k):
                L[k][j], L[p][j] = L[p][j], L[k][j]
            swaps += 1
        for i in range(k + 1, n):
            m = U[i][k] / U[k][k]
            L[i][k] = m
            U[i][k] = 0.0
            for j in range(k + 1, n):
                U[i][j] -= m * U[k][j]

    P = [[0.0] * n for _ in range(n)]
    for i in range(n):
        P[i][perm[i]] = 1.0
    return L, U, P, perm, swaps


def lu_solve(L, U, perm, b):
    n = len(b)
    y = [0.0] * n
    for i in range(n):
        s = b[perm[i]]
        for j in range(i):
            s -= L[i][j] * y[j]
        y[i] = s
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= U[i][j] * x[j]
        x[i] = s / U[i][i]
    return x


def lu_invert(L, U, perm, n):
    columns = []
    for i in range(n):
        e = [1.0 if k == i else 0.0 for k in range(n)]
        columns.append(lu_solve(L, U, perm, e))
    return transpose(columns)


def lu_det(U, swaps):
    det = 1.0
    for i in range(len(U)):
        det *= U[i][i]
    return -det if swaps % 2 else det


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/task1.txt"
    n, A, b = read_system(path)
    print(f"Входные данные: {path}\n")
    print_matrix(A, "Матрица A:")
    print_vector(b, "Вектор b:")

    L, U, P, perm, swaps = lu_decompose(A)

    print_matrix(L, "1) Матрица L (диагональ единичная):")
    print_matrix(U, "   Матрица U:")

    LU = mat_mul(L, U)
    PA = mat_mul(P, A)
    print_matrix(LU, "2) L * U:")
    print_matrix(PA, "7) P * A (для сравнения):")
    print(f"   ||L*U - P*A|| = {mat_max_abs_diff(LU, PA):.3e}\n")

    x = lu_solve(L, U, perm, b)
    print_vector(x, "3) Решение системы A*x = b:")
    r = [ri - bi for ri, bi in zip(mat_vec(A, x), b)]
    print(f"   ||A*x - b|| = {max(abs(t) for t in r):.3e}\n")

    A_inv = lu_invert(L, U, perm, n)
    print_matrix(A_inv, "4) Обратная матрица A^(-1):")

    print(f"5) det A = {lu_det(U, swaps):.6f}\n")

    check = mat_mul(A, A_inv)
    print_matrix(check, "6) Проверка A * A^(-1):")
    print(f"   ||A*A^(-1) - E|| = {mat_max_abs_diff(check, identity(n)):.3e}")


if __name__ == "__main__":
    main()