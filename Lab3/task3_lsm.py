"""
Лабораторная работа 3.3 — метод наименьших квадратов (МНК).
ВАРИАНТ 5: узлы 0.1 0.5 0.9 1.3 1.7 2.1;  y = -2.3026 ... 0.74194.

Строятся приближающие многочлены P1, P2, P3 степеней 1, 2, 3.
МНК минимизирует сумму квадратов ошибок
    S = SUM_i ( P(x_i) - y_i )^2.
Условие минимума dS/da_j = 0 приводит к нормальной системе
    SUM_{k=0..m} a_k * SUM_i x_i^{k+j} = SUM_i y_i * x_i^j,   j = 0..m,
которая решается методом Гаусса. Качество приближения измеряется
    MSE_m = S / n  (средний квадрат ошибки по всем исходным точкам).

Вход: data/task3.txt — узлы и значения.
Запуск: python task3_lsm.py [файл]
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import fmt_poly, poly_eval, read_task3, solve_linear, style_axes


def normal_system(xs, ys, m):
    # Нормальная система МНК для многочлена степени m
    n = len(xs)
    powers = [sum(x ** k for x in xs) for k in range(2 * m + 1)]
    rhs = [sum(y * x ** j for x, y in zip(xs, ys)) for j in range(m + 1)]
    A = [[powers[k + j] for k in range(m + 1)] for j in range(m + 1)]
    return A, rhs, n


def fit_poly(xs, ys, m):
    A, rhs, n = normal_system(xs, ys, m)
    coeffs = solve_linear(A, rhs)
    s = sum((poly_eval(coeffs, x) - y) ** 2 for x, y in zip(xs, ys))
    return coeffs, s, s / n


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/task3.txt"
    xs, ys = read_task3(path)
    n = len(xs)

    print("Лабораторная работа 3.3 — вариант 5")
    print("Исходные данные (n = %d точек):" % n)
    print("      i        x_i           y_i")
    for i, (x, y) in enumerate(zip(xs, ys)):
        print(f"     {i}   {x:8.4f}   {y:10.5f}")
    print()

    results = {}
    for m in (1, 2, 3):
        coeffs, s, mse = fit_poly(xs, ys, m)
        results[m] = (coeffs, s, mse)

    names = {1: "первой", 2: "второй", 3: "третьей"}
    for m in (1, 2, 3):
        coeffs, s, mse = results[m]
        print(f"{m}) Многочлен {names[m]} степени P_{m}(x) "
              f"(нормальная система {m + 1}x{m + 1}):")
        print("     коэффициенты: " +
              ", ".join(f"a{k} = {c:.6f}" for k, c in enumerate(coeffs)))
        print("     P_%d(x) = %s" % (m, fmt_poly(coeffs)))
        print(f"     Сумма квадратов ошибок S_{m} = {s:.6f};  "
              f"MSE_{m} = S_{m}/n = {mse:.6f}")
        print()

    print("6) Сравнение ошибок:")
    for m in (1, 2, 3):
        _, s, mse = results[m]
        print(f"     P_{m}:  S = {s:9.6f}   MSE = {mse:.6f}")
    best = min(results, key=lambda m: results[m][2])
    print(f"     Лучше всех приближает исходные данные многочлен "
          f"P_{best} (наименьший MSE = {results[best][2]:.6f});")
    print("     с ростом степени ошибка не возрастает, но усложняется модель —")
    print("     для табличных данных y = ln x оптимальный баланс точности и")
    print("     простоты даёт парабола P_2.\n")

    # графическая часть
    root = os.path.dirname(os.path.abspath(__file__))
    png = os.path.join(root, "plots", "task3_lsm.png")
    a, b = xs[0] - 0.1, xs[-1] + 0.1
    grid = [a + (b - a) * i / 400 for i in range(401)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    style_axes(ax)
    ax.plot(xs, ys, "o", color="#151513", ms=8, zorder=5,
            label="исходные точки")
    colors = {1: "#a48e4b", 2: "#a25b54", 3: "#3a5f8a"}
    styles = {1: "--", 2: "-", 3: "-."}
    for m in (1, 2, 3):
        coeffs = results[m][0]
        pv = [poly_eval(coeffs, x) for x in grid]
        ax.plot(grid, pv, color=colors[m], lw=2.0, linestyle=styles[m],
                label=f"P_{m}(x), MSE = {results[m][2]:.4f}")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Задание 3.3: МНК, многочлены 1-3 степени (вариант 5)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(png)
    print("График сохранён: plots/task3_lsm.png")


if __name__ == "__main__":
    main()