"""
Лабораторная работа 1.2 — метод прогонки (алгоритм Томаса).

Система: a_i·x_{i-1} + b_i·x_i + c_i·x_{i+1} = d_i,  i = 1..n,  a_1 = 0, c_n = 0.
Матрица НЕ хранится (она разреженная) — только три диагонали и правая часть.

Прямая прогонка (x_i = alpha_i·x_{i+1} + beta_i):
    alpha_1 = -c_1/b_1,                    beta_1 = d_1/b_1,
    alpha_i = -c_i/(a_i·alpha_{i-1} + b_i),    beta_i = (d_i - a_i·beta_{i-1})/(a_i·alpha_{i-1} + b_i).
Обратная прогонка: x_n = beta_n,  x_i = alpha_i·x_{i+1} + beta_i.

Вход: data/task2.txt: n, затем n строк по 4 числа "a_i b_i c_i d_i".
Запуск: python task2_tridiagonal.py [файл]
"""

import sys

from common import read_lines, parse_floats, print_vector, vec_norm2


def tridiagonal_solve(a, b, c, d):
    n = len(b)
    alpha = [0.0] * n
    beta = [0.0] * n

    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]
    for i in range(1, n):
        den = a[i] * alpha[i - 1] + b[i]
        if abs(den) < 1e-14:
            raise ZeroDivisionError(f"Шаг {i+1}: a·alpha + b ≈ 0 — прогонка не применима")
        alpha[i] = -c[i] / den
        beta[i] = (d[i] - a[i] * beta[i - 1]) / den

    x = [0.0] * n
    x[n - 1] = beta[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]
    return alpha, beta, x


def residual(a, b, c, x, d):
    n = len(x)
    r = []
    for i in range(n):
        s = b[i] * x[i]
        if i > 0:
            s += a[i] * x[i - 1]
        if i < n - 1:
            s += c[i] * x[i + 1]
        r.append(s - d[i])
    return r


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/task2.txt"
    lines = read_lines(path)
    n = int(parse_floats(lines[0])[0])
    a, b, c, d = [], [], [], []
    for i in range(n):
        row = parse_floats(lines[i + 1])
        if len(row) != 4:
            raise ValueError("Ожидается 4 числа: a_i b_i c_i d_i")
        a.append(row[0]); b.append(row[1]); c.append(row[2]); d.append(row[3])
    a[0] = 0.0
    c[n - 1] = 0.0

    print(f"Входные данные: {path}\n")
    print("  i      a_i        b_i        c_i        d_i")
    for i in range(n):
        print(f"  {i+1}  {a[i]:9.4f}  {b[i]:9.4f}  {c[i]:9.4f}  {d[i]:9.4f}")

    stable = all(abs(a[i]) + abs(c[i]) <= abs(b[i]) + 1e-12 for i in range(n))
    print("\nУсловие устойчивости |a_i|+|c_i| ≤ |b_i|:",
          "выполнено" if stable else "НЕ выполнено!")

    alpha, beta, x = tridiagonal_solve(a, b, c, d)

    print("\n1) Прогоночные коэффициенты (прямая прогонка):")
    print("   i      alpha_i       beta_i")
    for i in range(n):
        print(f"   {i+1}  {alpha[i]:11.6f}  {beta[i]:11.6f}")
    print("   (alpha_n не используется: c_n = 0)")

    print_vector(x, "2) Решение системы:")
    print(f"Проверка: ||A·x - d|| = {vec_norm2(residual(a, b, c, x, d)):.3e}")


if __name__ == "__main__":
    main()