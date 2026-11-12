"""
Лабораторная работа 3.1 — интерполяционные многочлены Лагранжа и Ньютона.
ВАРИАНТ 5:

    y = ln(x),  узлы Xi = 0.2, 0.6, 1.0, 1.4 (а) и б) совпадают),  x* = 0.8.

Схема решения:
  1) многочлен Лагранжа
         L3(x) = SUM y_i * PROD_{j!=i} (x - x_j) / (x_i - x_j),
     здесь он раскрыт в стандартный вид c0 + c1*x + c2*x^2 + c3*x^3;
  2) таблица разделённых разностей и многочлен Ньютона
         N3(x) = f(x0) + f(x0,x1)(x-x0) + f(x0,x1,x2)(x-x0)(x-x1)
                 + f(x0,...,x3)(x-x0)(x-x1)(x-x2);
  3) значения L3(x*), N3(x*), точное f(x*) = ln(x*), абсолютная погрешность
     интерполяции и разность результатов двух методов.
     Теоретическая погрешность:  R3(x) = f''''(xi)/4! * PROD (x - Xi).

Вход: data/task1.txt — узлы; x*.
Запуск: python task1_interp.py [файл]
"""

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (fmt_poly, poly_add, poly_eval, poly_from_root, poly_mul,
                    poly_scale, read_task1, style_axes)

# f(x) = ln(x) — функция известна аналитически
f = math.log


def lagrange_coeffs(xs, ys):
    # Раскрытие многочлена Лагранжа в стандартные коэффициенты
    n = len(xs)
    total = [0.0] * n
    for i in range(n):
        basis = [1.0]
        denom = 1.0
        for j in range(n):
            if j != i:
                basis = poly_mul(basis, poly_from_root(xs[j]))
                denom *= (xs[i] - xs[j])
        total = poly_add(total, poly_scale(basis, ys[i] / denom))
    return total


def newton_coeffs(xs, ys):
    # Коэффициенты при базисных слагаемых формы Ньютона = разделённые разности
    n = len(xs)
    # dd[k][i] = f[x_i, ..., x_{i+k}]
    dd = [[y for y in ys]]
    for k in range(1, n):
        prev = dd[k - 1]
        dd.append([(prev[i + 1] - prev[i]) / (xs[i + k] - xs[i])
                   for i in range(n - k)])
    return [col[0] for col in dd]        # f[x0], f[x0,x1], f[x0,x1,x2], ...


def divided_diff_table(xs, ys):
    # Полная треугольная таблица разделённых разностей
    n = len(xs)
    dd = [[y for y in ys]]
    for k in range(1, n):
        prev = dd[k - 1]
        dd.append([(prev[i + 1] - prev[i]) / (xs[i + k] - xs[i])
                   for i in range(n - k)])
    return dd


def newton_expand(xs, coeffs):
    # Раскрытие формы Ньютона: N(x) = c0 + c1*(x-x0) + ... + cn*(x-x0)...(x-xn-1)
    p = [coeffs[-1]]
    for k in range(len(coeffs) - 2, -1, -1):
        p = poly_add(poly_mul(p, poly_from_root(xs[k])), [coeffs[k]])
    return p


def print_table(xs, ys):
    print("1) Исходные узлы интерполяции и значения функции y_i = ln(x_i):")
    print("      i        x_i            y_i")
    for i, (x, y) in enumerate(zip(xs, ys)):
        print(f"     {i}   {x:8.4f}   {y:12.6f}")
    print()


def print_lagrange(xs, ys, coeffs, x_star):
    print("2) Интерполяционный многочлен Лагранжа (раскрытый вид):")
    print("   L3(x) = " + fmt_poly(coeffs))
    lv = poly_eval(coeffs, x_star)
    print(f"   L3(x*) = L3({x_star:g}) = {lv:.8f}\n")
    return lv


def print_newton(xs, ys, dd, coeffs, x_star):
    print("3) Таблица разделённых разностей:")
    head = "   x_i      y_i"
    for k in range(1, len(xs)):
        head += f"   f[...,x..x_{k}]  "
    print(head)
    n = len(xs)
    for i in range(n):
        row = f"  {xs[i]:6.3f}  {dd[0][i]:9.5f}"
        for k in range(1, n):
            if i < len(dd[k]):
                row += f"  {dd[k][i]:12.5f}"
            else:
                row += "             "
        print(row)
    print()
    nf = ""
    for k in range(len(coeffs)):
        term = f"{coeffs[k]:.5f}" if k == 0 else f"{abs(coeffs[k]):.5f}*"
        term += "*".join(
            f"(x-{xs[j]:g})" if xs[j] >= 0 else f"(x+{-xs[j]:g})"
            for j in range(k))
        if k == 0:
            nf = term
        else:
            nf += (" + " if coeffs[k] >= 0 else " - ") + term
    print("4) Интерполяционный многочлен Ньютона:")
    print("   N3(x) = " + nf)
    ex = newton_expand(xs, coeffs)
    print("   (в стандартном виде: N3(x) = " + fmt_poly(ex) + ")")
    nv = poly_eval(ex, x_star)
    print(f"   N3(x*) = N3({x_star:g}) = {nv:.8f}\n")
    return nv


def make_plot(xs, ys, x_star, l_val, coeffs, newt_coeffs, out_png):
    a, b = xs[0] - 0.05, xs[-1] + 0.05
    grid = [a + (b - a) * i / 400 for i in range(401)]
    fx = [f(x) for x in grid]
    lx = [poly_eval(coeffs, x) for x in grid]
    nx = [poly_eval(newt_coeffs, x) for x in grid]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    style_axes(ax)
    ax.plot(grid, fx, color="#3a5f8a", lw=2.2, label="f(x) = ln x")
    ax.plot(grid, lx, color="#a48e4b", lw=2.0, linestyle="--",
            label="многочлен Лагранжа L3(x)")
    ax.plot(grid, nx, color="#529067", lw=1.4, linestyle=":",
            label="многочлен Ньютона N3(x)")
    ax.plot(xs, ys, "o", color="#151513", ms=7, zorder=5, label="узлы (xi, yi)")
    ax.plot([x_star], [l_val], "D", color="#a25b54", ms=8, zorder=6,
            label="x* = %.1f" % x_star)
    ax.axvline(x_star, color="#a25b54", lw=0.8, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Задание 3.1: интерполяция ln x (вариант 5)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/task1.txt"
    xs, x_star = read_task1(path)
    ys = [f(x) for x in xs]

    print("Лабораторная работа 3.1 — вариант 5")
    print("Функция: y = ln(x);  узлы: " + ", ".join(f"{x:g}" for x in xs))
    print(f"Точка интерполяции x* = {x_star:g}\n")

    print_table(xs, ys)

    coeffs = lagrange_coeffs(xs, ys)
    l_val = print_lagrange(xs, ys, coeffs, x_star)

    dd = divided_diff_table(xs, ys)
    ndd = newton_coeffs(xs, ys)
    n_val = print_newton(xs, ys, dd, ndd, x_star)

    exact = f(x_star)
    print("5) Значение многочлена Лагранжа в заданной точке: "
          f"L3({x_star:g}) = {l_val:.8f}")
    print("6) Значение многочлена Ньютона в заданной точке: "
          f"N3({x_star:g}) = {n_val:.8f}")
    print("7) Точное значение функции: f(x*) = ln(%g) = %.8f" % (x_star, exact))
    err_l = abs(exact - l_val)
    err_n = abs(exact - n_val)
    print("8) Абсолютная погрешность интерполяции:")
    print("     |f(x*) - L3(x*)| = %.2e" % err_l)
    print("     |f(x*) - N3(x*)| = %.2e" % err_n)
    print("9) Разность результатов двух методов |L3(x*) - N3(x*)| = %.2e "
          "(совпадают с точностью машинной арифметики)" % abs(l_val - n_val))

    # Теоретический член погрешности: R3(x*) = f''''(xi)/4! * П(x*-xi),
    # xi неизвестна и лежит в (x0, x3); при xi ~ x* оценка совпадает по порядку
    prod = 1.0
    for x in xs:
        prod *= (x_star - x)
    est = 6.0 / x_star ** 4 / 24.0 * abs(prod)
    print("     Теория: R3(x*) = f''''(xi)/4!*П(x*-xi); при xi ~ x* "
          "получаем ~%.2e —" % est)
    print("     по порядку совпадает с фактической погрешностью %.2e" % err_l)

    # графическая часть
    root = os.path.dirname(os.path.abspath(__file__))
    png = os.path.join(root, "plots", "task1_interp.png")
    make_plot(xs, ys, x_star, l_val, coeffs, newton_expand(xs, ndd), png)
    print("\nГрафик сохранён: plots/task1_interp.png")


if __name__ == "__main__":
    main()
