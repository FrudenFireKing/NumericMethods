"""
Лабораторная работа 3.2 — построение кубического сплайна.
ВАРИАНТ 5: узлы 0.1 0.5 0.9 1.3 1.7;  y = -2.3026 ... 0.53063;  x* = 0.8.

Граничные условия: нулевая кривизна на концах (естественный сплайн),
    S''(x0) = S''(xn) = 0.

На каждом участке [x_i, x_{i+1}] сплайн ищется в виде (t = x - x_i)
    S_i(x) = a_i + b_i*t + (c_i/2)*t^2 + (d_i/6)*t^3,
где a_i = y_i, c_i = S''(x_i). Коэффициенты c_i (i = 1..n-1) находятся
из классической трёхдиагональной системы
    h_{i-1} c_{i-1} + 2(h_{i-1}+h_i) c_i + h_i c_{i+1} =
                          3[ (y_{i+1}-y_i)/h_i - (y_i-y_{i-1})/h_{i-1} ],
с c_0 = c_n = 0 (прогонка); далее
    d_i = (c_{i+1} - c_i) / h_i,
    b_i = (y_{i+1} - y_i)/h_i - h_i*(2 c_i + c_{i+1}) / 6.
Тогда S(x) непрерывна вместе с S' и S'' на всём отрезке, S''(x_i) = c_i.

Вход: data/task2.txt — узлы, значения, x*.
Запуск: python task2_spline.py [файл]
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import read_task2, style_axes


def solve_tridiagonal(a, b, c, d):
    # Прогонка для системы с лентой (a-нижняя, b-главная, c-верхняя диагональ)
    n = len(b)
    p = [0.0] * n
    q = [0.0] * n
    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] + a[i] * p[i - 1]
        p[i] = -c[i] / denom if i < n - 1 else 0.0
        q[i] = (d[i] - a[i] * q[i - 1]) / denom
    x = [0.0] * n
    x[n - 1] = q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]
    return x


def build_spline(xs, ys):
    # Коэффициенты a, b, c, d для каждого участка
    n = len(xs) - 1                      # число участков
    h = [xs[i + 1] - xs[i] for i in range(n)]

    # трёхдиагональная система для c_1..c_{n-1}; c_0 = c_n = 0
    m = n - 1
    A = [0.0] * m
    B = [0.0] * m
    C = [0.0] * m
    D = [0.0] * m
    for k in range(m):
        i = k + 1                        # внутренний узел
        if k > 0:
            A[k] = h[i - 1]
        B[k] = 2.0 * (h[i - 1] + h[i])
        if k < m - 1:
            C[k] = h[i]
        D[k] = 3.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1])
    c_inner = solve_tridiagonal(A, B, C, D)

    c = [0.0] + list(c_inner) + [0.0]
    a = list(ys[:n])
    d = [(c[i + 1] - c[i]) / h[i] for i in range(n)]
    b = [(ys[i + 1] - ys[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 6.0
         for i in range(n)]
    return a, b, c, d, h


def spline_eval(xs, a, b, c, d, x):
    # Поиск интервала и вычисление S(x); возвращает (S, i)
    n = len(xs) - 1
    if x < xs[0] or x > xs[-1]:
        raise ValueError("x* вне области интерполяции")
    i = n - 1
    for k in range(n):
        if xs[k] <= x <= xs[k + 1]:
            i = k
            break
    t = x - xs[i]
    s = a[i] + b[i] * t + 0.5 * c[i] * t * t + d[i] * t * t * t / 6.0
    return s, i


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/task2.txt"
    xs, ys, x_star = read_task2(path)
    a, b, c, d, h = build_spline(xs, ys)
    n = len(xs) - 1

    print("Лабораторная работа 3.2 — вариант 5")
    print("Кубический сплайн с нулевой кривизной на концах "
          "(S''(x0) = S''(xn) = 0)")
    print("1) Исходные узлы и значения функции:")
    print("      i        x_i           y_i        h_i")
    for i in range(n + 1):
        hi = f"{h[i]:.4f}" if i < n else "  —"
        print(f"     {i}   {xs[i]:8.4f}   {ys[i]:10.5f}    {hi}")
    print()

    print("2) Коэффициенты сплайна (t = x - x_i):")
    print("    S_i(x) = a_i + b_i*t + (c_i/2)*t^2 + (d_i/6)*t^3")
    print("      i     a_i          b_i          c_i          d_i")
    for i in range(n):
        print(f"     {i}  {a[i]:10.5f}  {b[i]:11.5f}  {c[i]:11.5f}  "
              f"{d[i]:11.5f}")
    print()

    print("3) Интервалы действия каждого участка:")
    for i in range(n):
        print(f"     S_{i}(x) действует на [{xs[i]:.4f}, {xs[i + 1]:.4f}]")
    print()

    s_val, idx = spline_eval(xs, a, b, c, d, x_star)
    print("4) Значение сплайна в заданной точке:")
    print(f"     x* = {x_star:g}  ->  S(x*) = {s_val:.8f}")
    print("5) Номер интервала, содержащего x*: i = %d, "
          "интервал [%.4f, %.4f]" % (idx, xs[idx], xs[idx + 1]))
    print()

    print("6) Проверка прохождения сплайна через исходные узлы:")
    ok = True
    for i, (x, y) in enumerate(zip(xs, ys)):
        s_i, _ = spline_eval(xs, a, b, c, d, x)
        gap = abs(s_i - y)
        ok = ok and gap < 1e-6
        print(f"     S({x:.4f}) = {s_i:10.5f},  y_i = {y:10.5f},  "
              f"|S - y| = {gap:.2e}")
    print("     Все узлы проходят: " + ("да" if ok else "НЕТ") + "\n")

    print("7) Вторая производная на концах (проверка граничных условий):")
    print(f"     S''(x0) = c_0 = {c[0]:.6f}   (требуется 0)")
    print(f"     S''(xn) = c_n = {c[n]:.6f}   (требуется 0)\n")

    # графическая часть
    root = os.path.dirname(os.path.abspath(__file__))
    png = os.path.join(root, "plots", "task2_spline.png")
    shades = ["#2d5f8a", "#3a77a8", "#4a8ec4", "#63a4d4"][:n]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    style_axes(ax)
    for i in range(n):
        g = [xs[i] + (xs[i + 1] - xs[i]) * t / 60 for t in range(61)]
        sv = [a[i] + b[i] * (x - xs[i]) + 0.5 * c[i] * (x - xs[i]) ** 2 +
              d[i] * (x - xs[i]) ** 3 / 6.0 for x in g]
        ax.plot(g, sv, color=shades[i], lw=2.2,
                label=f"участок {i} [{xs[i]:g}, {xs[i + 1]:g}]")
    ax.plot(xs, ys, "o", color="#151513", ms=8, zorder=5, label="узлы")
    ax.plot([x_star], [s_val], "D", color="#a25b54", ms=9, zorder=6,
            label="x* = %.1f, S = %.4f" % (x_star, s_val))
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_title("Задание 3.2: кубический сплайн (вариант 5)", fontsize=11)
    fig.tight_layout()
    fig.savefig(png)
    print("График сохранён: plots/task2_spline.png")


if __name__ == "__main__":
    main()