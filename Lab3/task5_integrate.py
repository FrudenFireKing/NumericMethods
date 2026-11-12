"""
Лабораторная работа 3.5 — численное интегрирование и уточнение по
Рунге-Ромбергу. ВАРИАНТ 5:

    F = INT_{-1}^{1} dx / ((2x+7)(3x+4)),   h1 = 0.5,  h2 = 0.25.

Реализованы: левые прямоугольники, правые прямоугольники, средние
прямоугольники, трапеции, Симпсон. Порядки точности p: 1, 1, 2, 2, 4.

Правило Рунге-Ромберга: если I(h1) и I(h2) — результаты одного метода
с шагами h1, h2 = h1/2, то уточнённое значение
    I_RR = I(h2) + ( I(h2) - I(h1) ) / (r^p - 1),   r = h1/h2 = 2,
а оценка погрешности результата с шагом h2:
    R ~ | I(h2) - I(h1) | / (r^p - 1).

Точное значение (частные дроби):
    1/((2x+7)(3x+4)) = (-2/13)/(2x+7) + (3/13)/(3x+4),
    F = (1/13) * ln(35/9).

Вход: data/task5.txt — пределы интегрирования, шаги h1, h2.
Запуск: python task5_integrate.py [файл]
"""

import math
import sys

from common import read_task5


def y(x):
    return 1.0 / ((2.0 * x + 7.0) * (3.0 * x + 4.0))


def left_rect(a, b, n):
    h = (b - a) / n
    return h * sum(y(a + i * h) for i in range(n))


def right_rect(a, b, n):
    h = (b - a) / n
    return h * sum(y(a + i * h) for i in range(1, n + 1))


def mid_rect(a, b, n):
    h = (b - a) / n
    return h * sum(y(a + (i + 0.5) * h) for i in range(n))


def trapezoid(a, b, n):
    h = (b - a) / n
    s = 0.5 * (y(a) + y(b)) + sum(y(a + i * h) for i in range(1, n))
    return h * s


def simpson(a, b, n):
    if n % 2 == 1:
        raise ValueError("метод Симпсона требует чётного числа участков")
    h = (b - a) / n
    s_odd = sum(y(a + i * h) for i in range(1, n, 2))
    s_even = sum(y(a + i * h) for i in range(2, n, 2))
    return h / 3.0 * (y(a) + y(b) + 4.0 * s_odd + 2.0 * s_even)


def runge_romberg(i1, i2, p):
    # Уточнение по Рунге-Ромбергу: h2 = h1/2, r = 2
    r = 2.0
    refined = i2 + (i2 - i1) / (r ** p - 1.0)
    err_est = abs(i2 - i1) / (r ** p - 1.0)
    return refined, err_est


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/task5.txt"
    a, b, h1, h2 = read_task5(path)
    n1 = int(round((b - a) / h1))
    n2 = int(round((b - a) / h2))

    print("Лабораторная работа 3.5 — вариант 5")
    print(f"Интеграл: F = INT_{a:g}^{b:g} dx / ((2x+7)(3x+4))")
    print(f"Шаги: h1 = {h1:g} (n1 = {n1} участков),  h2 = {h2:g} "
          f"(n2 = {n2} участков)\n")

    exact = math.log(35.0 / 9.0) / 13.0
    print(f"Точное значение (разложение на частные дроби): "
          f"F = (1/13)*ln(35/9) = {exact:.8f}\n")

    methods = [
        ("Левые прямоугольники",   left_rect,  1),
        ("Правые прямоугольники",  right_rect, 1),
        ("Средние прямоугольники", mid_rect,   2),
        ("Трапеции",               trapezoid,  2),
        ("Симпсон",                simpson,    4),
    ]

    header = (f"{'Метод':<24}{'I(h1)':>12}{'I(h2)':>12}{'p':>3}"
              f"{'I_Рунге-Ромб':>15}{'R(h2) оценка':>14}{'|I(h2)-F|':>12}")
    print("Результаты (таблица):")
    print(header)
    print("-" * len(header))

    best_name, best_err = None, float("inf")
    for name, fn, p in methods:
        i1 = fn(a, b, n1)
        i2 = fn(a, b, n2)
        rr, err_est = runge_romberg(i1, i2, p)
        real_err = abs(i2 - exact)
        print(f"{name:<24}{i1:>12.6f}{i2:>12.6f}{p:>3}"
              f"{rr:>15.6f}{err_est:>14.2e}{real_err:>12.2e}")
        if real_err < best_err:
            best_err, best_name = real_err, name
    print("-" * len(header))
    print()

    print("Абсолютные погрешности с шагом h1 (для сравнения):")
    for name, fn, p in methods:
        print(f"     {name:<24} |I(h1) - F| = {abs(fn(a, b, n1) - exact):.2e}")
    print()
    print(f"Наиболее точный результат на шаге h2 дал метод: {best_name} "
          f"(|ошибка| = {best_err:.2e}).")
    print("Это ожидаемо: у метода Симпсона наивысший порядок точности p = 4,")
    print("а для гладкой функции уточнённое значение I_Рунге-Ромберга близко"
          " к точному лучше всего.")


if __name__ == "__main__":
    main()