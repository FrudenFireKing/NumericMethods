"""
Лабораторная работа 1.4 — метод вращений (метод Якоби) для СИММЕТРИЧНОЙ матрицы.

На каждом шаге берётся максимальный по модулю внедиагональный элемент a_pq
и строится поворот G в плоскости (p, q), зануляющий его: T' = Gᵀ*T*G.
Угол:  τ = (a_qq - a_pp)/(2a_pq),  t = sign(τ)/(|τ|+√(1+τ²)),  c = 1/√(1+t²),  s = t*c.
V = произведение всех поворотов — матрица собственных векторов (столбцы),
диагональ итоговой T — собственные значения. Сходимость квадратичная.

Вход: data/task4.txt: n; симметричная матрица; eps. Запуск: python task4_jacobi.py [файл]
"""

import math
import sys

from common import (read_matrix_eps, identity, mat_mul, print_matrix,
                    print_vector, mat_max_abs_diff)


def max_offdiag(A):
    n = len(A)
    p, q = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j]) > abs(A[p][q]):
                p, q = i, j
    return p, q


def offdiag_norm(A):
    n = len(A)
    return math.sqrt(sum(A[i][j] ** 2
                         for i in range(n) for j in range(n) if i != j))


def jacobi_rotation_method(A, eps, max_rotations=100):
    n = len(A)
    if n == 1:
        return [A[0][0]], identity(1), []
    T = [row[:] for row in A]
    V = identity(n)
    history = []

    for it in range(1, max_rotations + 1):
        p, q = max_offdiag(T)
        if abs(T[p][q]) < eps:
            return [T[i][i] for i in range(n)], V, history

        apq = T[p][q]
        tau = (T[q][q] - T[p][p]) / (2.0 * apq)
        t = math.copysign(1.0, tau) / (abs(tau) + math.sqrt(1.0 + tau * tau))
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c

        for k in range(n):
            tkp, tkq = T[k][p], T[k][q]
            T[k][p] = c * tkp - s * tkq
            T[k][q] = s * tkp + c * tkq
        for k in range(n):
            tpk, tqk = T[p][k], T[q][k]
            T[p][k] = c * tpk - s * tqk
            T[q][k] = s * tpk + c * tqk
        T[p][q] = T[q][p] = 0.0

        for k in range(n):
            vkp, vkq = V[k][p], V[k][q]
            V[k][p] = c * vkp - s * vkq
            V[k][q] = s * vkp + c * vkq

        history.append((it, p + 1, q + 1, abs(apq), offdiag_norm(T)))

    raise RuntimeError("Метод вращений не сошёлся")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/task4.txt"
    n, A, eps = read_matrix_eps(path)

    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > 1e-9:
                raise ValueError("Матрица не симметрична")

    print(f"Входные данные: {path}")
    print(f"1) Заданная точность ε = {eps:g}\n")

    values, V, history = jacobi_rotation_method(A, eps)

    print("Зависимость погрешности от числа вращений:")
    print("  №    (p,q)    max|a_pq| до      ||T_off|| после")
    for it, p, q, mx, off in history:
        print(f"  {it:2d}   ({p},{q})   {mx:.6e}     {off:.6e}")
    print(f"  Всего вращений: {len(history)}\n")

    order = sorted(range(n), key=lambda i: -values[i])
    values = [values[i] for i in order]
    V = [[row[i] for i in order] for row in V]

    print("2) Найденные собственные значения:")
    for i, lam in enumerate(values, 1):
        print(f"   λ{i} = {lam:.6f}")

    print("\n3) Найденные собственные векторы:")
    for i in range(n):
        print_vector([V[r][i] for r in range(n)],
                     f"   v{i+1} (λ = {values[i]:.6f}):")
    print_matrix(V, "   Матрица собственных векторов V (столбцы — векторы):")

    Lam = [[values[i] if i == j else 0.0 for j in range(n)] for i in range(n)]
    AV = mat_mul(A, V)
    VLa = mat_mul(V, Lam)
    print_matrix(AV, "4) Проверка A*V:")
    print_matrix(VLa, "   Проверка V*Λ:")
    print(f"   ||A*V - V*Λ|| = {mat_max_abs_diff(AV, VLa):.3e}")


if __name__ == "__main__":
    main()