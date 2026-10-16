"""
Лабораторная работа 1.3 — метод простых итераций и метод Зейделя.

Оба метода решают СЛАУ A*x = b, записанную в форме x = B*x + c:
  МПИ (Якоби):  x_i^{k+1} = (b_i - Σ_{j!=i} a_ij*x_j^k) / a_ii  — только старые значения;
  Зейдель:      то же, но компоненты с номерами j < i берутся уже из ТЕКУЩЕЙ итерации.
Критерий останова: ||x^{k+1} - x^k||_inf < eps.

Вход: data/task3.txt: n; матрица A; вектор b; eps. Запуск: python task3_iterations.py [файл]
"""

import sys

from common import read_lines, parse_floats, mat_vec, print_vector, vec_max_abs_diff
from task1_lu import lu_decompose, lu_solve     # эталонное решение через LU из 1.1

MAX_ITER = 100000


def simple_iteration(A, b, eps):
    n = len(A)
    x = [0.0] * n
    for it in range(1, MAX_ITER + 1):
        nx = [0.0] * n
        for i in range(n):
            s = b[i]
            for j in range(n):
                if j != i:
                    s -= A[i][j] * x[j]
            nx[i] = s / A[i][i]
        if max(abs(nx[i] - x[i]) for i in range(n)) < eps:
            return nx, it
        x = nx
    raise RuntimeError("Метод простых итераций не сошёлся")


def zeidel(A, b, eps):
    n = len(A)
    x = [0.0] * n
    for it in range(1, MAX_ITER + 1):
        nx = x[:]
        for i in range(n):
            s = b[i]
            for j in range(n):
                if j != i:
                    s -= A[i][j] * nx[j]
            nx[i] = s / A[i][i]
        if max(abs(nx[i] - x[i]) for i in range(n)) < eps:
            return nx, it
        x = nx
    raise RuntimeError("Метод Зейделя не сошёлся")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/task3.txt"
    lines = read_lines(path)
    n = int(parse_floats(lines[0])[0])
    A = [parse_floats(lines[i + 1]) for i in range(n)]
    b = parse_floats(lines[n + 1])
    eps = parse_floats(lines[n + 2])[0]

    if any(abs(A[i][i]) < 1e-14 for i in range(n)):
        raise ValueError("Нулевой элемент на диагонали - методы неприменимы")

    print(f"Входные данные: {path}")
    print(f"1) Заданная точность eps = {eps:g}\n")

    # проверка условий сходимости
    dominant = all(abs(A[i][i]) > sum(abs(A[i][j]) for j in range(n) if j != i)
                   for i in range(n))
    q = max(sum(abs(A[i][j] / A[i][i]) for j in range(n) if j != i)
            for i in range(n))                       # ||B||_inf матрицы перехода
    print(f"Диагональное преобладание: {'есть' if dominant else 'нет'}")
    print(f"||B||_inf = {q:.6f} - " +
          ("достаточное условие сходимости выполнено" if q < 1
           else "сходимость НЕ гарантирована") + "\n")

    x1, it1 = simple_iteration(A, b, eps)
    x2, it2 = zeidel(A, b, eps)

    print_vector(x1, f"Метод простых итераций, {it1} итераций:")
    print_vector(x2, f"Метод Зейделя, {it2} итераций:")

    L, U, P, perm, _ = lu_decompose(A)   # эталон из задания 1.1
    x_ref = lu_solve(L, U, perm, b)
    print_vector(x_ref, "Эталонное решение (LU-разложение):")

    print(f"Погрешность МПИ:     ||x - x*|| = {vec_max_abs_diff(x1, x_ref):.3e}")
    print(f"Погрешность Зейделя: ||x - x*|| = {vec_max_abs_diff(x2, x_ref):.3e}")

    print("\n4) Сравнение по скорости сходимости:")
    if it2 < it1:
        print(f"   Зейдель: {it2} итер. < МПИ: {it1} итер. -> Зейдель сходится быстрее,")
        print("   т.к. сразу использует уже обновлённые компоненты (спектральный радиус меньше).")
    elif it2 > it1:
        print(f"   МПИ сошёлся быстрее: {it1} против {it2}.")
    else:
        print("   Одинаковое число итераций.")


if __name__ == "__main__":
    main()