"""
Лабораторная работа 1.5 — QR-разложение матрицы и QR-алгоритм поиска
собственных значений.

Часть 1. QR-разложение методом отражений Хаусхолдера.
  Для k = 0..n-2 берётся подстолбец x = (a_kk,...,a_nk) и строится матрица
  отражения  H = I - 2·v·vᵀ/||v||²,  v = x - alpha·e_1,  alpha = -sign(x_1)·||x||.
  Тогда H·x = alpha·e₁ — поддиагональная часть столбца зануляется. H симметрична
  и ортогональна (H·Hᵀ = E). После n-1 отражений:
      R = H_{n-2}···H₀·A  — верхнетреугольная,
      Q = H₀···H_{n-2}    — ортогональная,  A = Q·R.
  Знак alpha противоположен знаку x₁, чтобы не вычитать близкие числа.

Часть 2. QR-алгоритм полной проблемы собственных значений.
  Шаг: A_k = Q_k·R_k -> A_{k+1} = R_k·Q_k. Это ортогональное подобие
  (A_{k+1} = Q_kᵀ·A_k·Q_k), спектр сохраняется. Поддиагональный элемент,
  отделяющий группу собственных значений, убывает как |λ_{i+1}/λ_i|^k,
  поэтому итерации ведутся с дефляцией: как только |a[i][i-1]| < ε,
  матрица распадается на независимые блоки:
    * блок 1×1 -> вещественное собственное значение;
    * блок 2×2 -> корни λ² - (a+d)λ + (ad-bc) = 0; при disc < 0 это
      комплексно-сопряжённая пара (a+d)/2 ± i·√(-disc)/2.
  Итог — вещественная квазитреугольная форма матрицы.

Часть 3. Собственный пример матрицы с комплексно-сопряжёнными собственными
  значениями: A = [[5,-4,2],[10,-5,0],[0,-4,7]].
  Её характеристический многочлен λ³ - 7λ² + 15λ - 25 = (λ-5)(λ²-2λ+5),
  т.е. собственные значения λ = 5 и λ = 1 ± 2i.
  Подключаем numpy, чтобы проверить np.linalg.eigvals.

Вход: data/task5.txt и data/task5_complex.txt (n; матрица; ε).
Запуск: python task5_qr.py [файл] [файл_комплексного_примера]
"""

import math
import sys

from common import (read_matrix_eps, identity, mat_mul, transpose,
                    mat_max_abs_diff, print_matrix)

TINY = 1e-14      # порог нулевого столбца в методе Хаусхолдера


#  QR-разложение 

def qr_householder(M):
    #Разложение M = Q·R отражениями Хаусхолдера
    n = len(M)
    R = [row[:] for row in M]
    Q = identity(n)
    for k in range(n - 1):
        x = [R[i][k] for i in range(k, n)]  # зануляемая часть столбца
        norm_x = math.sqrt(sum(t * t for t in x))
        if norm_x < TINY:  # столбец уже нулевой
            continue
        alpha = -math.copysign(norm_x, x[0])  # куда отражаем столбец
        v = [x[0] - alpha] + x[1:] # v = x - alpha·e₁
        vv = sum(t * t for t in v)
        # R ← H·R,  H = I - 2·v·vᵀ/||v||²
        for j in range(n):
            s = sum(v[i] * R[k + i][j] for i in range(len(v)))
            for i in range(len(v)):
                R[k + i][j] -= 2.0 * s * v[i] / vv
        # Q ← Q·H (накопление произведения отражений)
        for i in range(n):
            s = sum(Q[i][k + t] * v[t] for t in range(len(v)))
            for t in range(len(v)):
                Q[i][k + t] -= 2.0 * s * v[t] / vv
    return Q, R


#  QR-алгоритм собственных значений

def block_eigenvalues_2x2(a, b, c, d):
    #Собственные значения блока [[a,b],[c,d]] через квадратное уравнение.#
    tr, det = a + d, a * d - b * c
    disc = tr * tr - 4.0 * det
    if disc >= 0.0:
        sq = math.sqrt(disc)
        return [0.5 * (tr + sq), 0.5 * (tr - sq)]
    re, im = 0.5 * tr, 0.5 * math.sqrt(-disc)
    return [complex(re, im), complex(re, -im)]


def qr_eigenvalues(A, eps, max_iter=100000):
    # QR-алгоритм с дефляцией -> (список λ, число QR-итераций)
    n = len(A)
    T = [row[:] for row in A]     # рабочая копия -> квазитреугольный вид
    eigs = []
    iters = 0
    stack = [(0, n)]      # окна [lo, hi) необработанных блоков
    while stack:
        lo, hi = stack.pop()
        if hi - lo == 1:
            eigs.append(T[lo][lo])
            continue
        # ищем расщепление: малый поддиагональный элемент (снизу вверх)
        split = None
        for i in range(hi - 1, lo, -1):
            if abs(T[i][i - 1]) < eps:
                split = i
                break
        if split is not None:        # блок распался на два независимых
            stack.append((lo, split))
            stack.append((split, hi))
            continue
        if hi - lo == 2:             # 2×2 решаем точно (возможна λ-пара)
            eigs.extend(block_eigenvalues_2x2(T[lo][lo], T[lo][lo + 1],
                                              T[lo + 1][lo], T[lo + 1][lo + 1]))
            continue
        # одна QR-итерация над активным окном: T[lo:hi] ← R·Q
        sub = [T[i][lo:hi] for i in range(lo, hi)]
        Q, R = qr_householder(sub)
        RQ = mat_mul(R, Q)
        for i in range(lo, hi):
            T[i][lo:hi] = RQ[i - lo]
        iters += 1
        stack.append((lo, hi))       # вернуть окно в очередь
        if iters > max_iter:
            raise RuntimeError("QR-алгоритм не сошёлся")
    return eigs, iters


# отчёт

def determinant(A):
    #Определитель через LU из задания 1.1 (для проверки Πλ).#
    from task1_lu import lu_decompose, lu_det
    _, U, _, _, swaps = lu_decompose(A)
    return lu_det(U, swaps)


def qr_report(path, header):
    #Пункты 1–4 задания для одной матрицы.#
    n, A, eps = read_matrix_eps(path)
    print(header)
    print(f"Файл: {path}    Заданная точность ε = {eps:g}\n")

    print_matrix(A, "1) Исходная матрица A:")

    Q, R = qr_householder(A)
    print_matrix(Q, "2) Матрица Q:")
    print_matrix(R, "   Матрица R:")

    QR = mat_mul(Q, R)
    print_matrix(QR, "3) Произведение Q·R:")
    print(f"   ||Q·R - A||  = {mat_max_abs_diff(QR, A):.3e}")
    print(f"   ||Qᵀ·Q - E|| = "
          f"{mat_max_abs_diff(mat_mul(transpose(Q), Q), identity(n)):.3e}\n")

    eigs, iters = qr_eigenvalues(A, eps)
    print(f"4) Собственные значения (QR-алгоритм, {iters} QR-итераций):")
    for z in sorted(eigs, key=lambda z: (-complex(z).real, -complex(z).imag)):
        if isinstance(z, complex):
            print(f"   λ = {z.real:+.6f} {z.imag:+.6f}i")
        else:
            print(f"   λ = {z:+.6f}")

    tr = sum(A[i][i] for i in range(n))
    s = sum(complex(z) for z in eigs)
    p = complex(1.0)
    for z in eigs:
        p *= complex(z)
    print(f"   Проверка: Σλ = {s.real:.6f} (след матрицы {tr:.6f}), "
          f"Πλ = {p.real:.6f} (det A = {determinant(A):.6f})")
    return A, eps, eigs


def numpy_check(A, eigs):
    #Проверка через NumPy — разрешено ТОЛЬКО для комплексного примера.#
    import numpy as np
    ref = sorted(np.linalg.eigvals(np.asarray(A, dtype=float)).tolist(),
                 key=lambda z: (z.real, z.imag))
    ours = sorted((complex(z) for z in eigs), key=lambda z: (z.real, z.imag))
    diff = max(abs(o - r) for o, r in zip(ours, ref))
    print("\n   Проверка NumPy (np.linalg.eigvals):")
    for z in ref:
        print(f"      λ = {z.real:+.6f} {z.imag:+.6f}i")
    print(f"   max |λ(наш) - λ(NumPy)| = {diff:.3e}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/task5.txt"
    path_c = sys.argv[2] if len(sys.argv) > 2 else "data/task5_complex.txt"

    qr_report(path, "Задание 1.5 — вариант 5\n" + "─" * 60)

    print("\n" + "=" * 60)
    print("Собственный пример: комплексно-сопряжённые собственные значения")
    print("Матрица построена из λ³ - 7λ² + 15λ - 25 = (λ-5)(λ²-2λ+5),")
    print("т.е. собственные значения: 5 и 1 ± 2i")
    print("=" * 60 + "\n")
    A2, _, eigs2 = qr_report(path_c, "")
    numpy_check(A2, eigs2)


if __name__ == "__main__":
    main()
