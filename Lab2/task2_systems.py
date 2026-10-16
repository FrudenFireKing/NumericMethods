"""
Лабораторная работа 2.2 — методы простой итерации и Ньютона
для систем нелинейных уравнений.  ВАРИАНТ 5, параметр a = 2:

    x1 - cos(x2)    = 1
    x2 - lg(x1 + 1) = a          (lg — десятичный логарифм)

Область определения системы: x1 > -1.

Схема решения:
  1) графическое отделение: '#' — кривая первого уравнения
     (x1 = 1 + cos(x2)), 'o' — второго (x2 = a + lg(x1 + 1));
     начальное приближение x0 (по пересечению '*') задаётся в файле;
  2) автопоиск всех пересечений кривых + уточнение Ньютоном —
     перечень всех решений; по условию задачи выбирается решение
     с положительными компонентами (здесь оно единственное);
  3) МПИ в естественной форме x = Phi(x):
         x1 = 1 + cos(x2),   x2 = a + lg(x1 + 1)
     (система уже разрешена относительно каждой переменной);
     критерий останова ||x_(k+1) - x_k||_inf < eps;
  4) анализ сходимости.  Phi'(x) = [[0, -sin(x2)], [1/((x1+1)ln10), 0]],
     на решении: ||Phi'||_inf = 0.832 < 1 (достаточное условие),
     собственные значения lambda1,2 = ±0.5i, rho = 0.5 < 1
     (необходимое и достаточное) — сходимость линейная КОЛЕБАТЕЛЬНАЯ:
     диагональ Phi' нулевая, поэтому ошибка за два шага умножается на
     -|l1*l2| = -0.25, т.е. ||e_(k+2)|| = 0.25*||e_k||.
     Метод Ньютона (J*Delta = -F) сходится квадратично: det J(x*) = 1.25.

Вход: data/task22.txt — a; окно графика по x1 (низ обязан быть > -1);
по x2; eps; x0_МПИ; x0_Ньютон.  Запуск: python task2_systems.py [файл]
"""

import cmath
import math
import sys

from common import read_task2, vec_sub, vec_inf, solve_linear

MAX_ITER = 1000
LN10 = math.log(10.0)


# СИСТЕМА ВАРИАНТА 5 (a = 2): F, J, Phi, dPhi

def F(x, a):
    # Вектор левых частей: F1 = x1 - cos(x2) - 1, F2 = x2 - lg(x1+1) - a
    return [x[0] - math.cos(x[1]) - 1.0,
            x[1] - math.log10(x[0] + 1.0) - a]


def J(x, a):
    # Матрица Якоби dF/dx (для метода Ньютона)
    return [[1.0, math.sin(x[1])],
            [-1.0 / ((x[0] + 1.0) * LN10), 1.0]]


def Phi(x, a):
    # Итерационная форма МПИ: x1 = 1 + cos(x2), x2 = a + lg(x1 + 1)
    if x[0] + 1.0 <= 0.0:
        raise ValueError("выход из области определения (x1 <= -1)")
    return [1.0 + math.cos(x[1]),
            a + math.log10(x[0] + 1.0)]


def dPhi(x, a):
    # Якобиан итерационной формы — для проверки условия сходимости
    return [[0.0, -math.sin(x[1])],
            [1.0 / ((x[0] + 1.0) * LN10), 0.0]]




# Методы

def simple_iteration(x0, a, eps, max_iter=MAX_ITER):
# МПИ для систем: x_(k+1) = Phi(x_k); останов ||dx||_inf < eps
    xs = [x0[:]]
    for k in range(1, max_iter + 1):
        try:
            x = Phi(xs[-1], a)
        except ValueError as exc:
            raise RuntimeError(f"МПИ прерван на итерации {k}: {exc}") from None
        xs.append(x)
        if vec_inf(vec_sub(x, xs[-2])) < eps:
            return xs, k
    raise RuntimeError(f"МПИ не сошёлся за {max_iter} итераций — "
                       f"по-видимому, rho(Phi') >= 1")


def newton(x0, a, eps, max_iter=MAX_ITER):
# Ньютон для систем: J(x_k)*Delta_k = -F(x_k), x_(k+1) = x_k + Delta_k
    xs = [x0[:]]
    for k in range(1, max_iter + 1):
        xk = xs[-1]
        try:
            delta = solve_linear(J(xk, a), [-fi for fi in F(xk, a)])
        except ValueError as exc:
            raise RuntimeError(f"Ньютон прерван на итерации {k}: {exc}") from None
        x = [xi + di for xi, di in zip(xk, delta)]
        xs.append(x)
        if vec_inf(delta) < eps:
            return xs, k
    raise RuntimeError(f"Метод Ньютона не сошёлся за {max_iter} итераций")


# Проверка условия сходимости МПИ

def eig2(M):
    # Собственные значения матрицы 2x2 (могут быть комплексными)
    tr = M[0][0] + M[1][1]
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    d = cmath.sqrt(tr * tr - 4.0 * det)
    return [(tr + d) / 2.0, (tr - d) / 2.0]


def matrix_inf_norm(M):
    return max(sum(abs(v) for v in row) for row in M)


# График и поиск всех решений

def ascii_plot(a, x1r, x2r, width=69, height=23, nscan=600):
    # Кривые F1 = 0 ('#') и F2 = 0 ('o'); '*' — обе в одной клетке
    x1min, x1max = x1r
    x2min, x2max = x2r
    grid = [[" "] * width for _ in range(height)]

    def col(x1):
        return round((x1 - x1min) / (x1max - x1min) * (width - 1))

    def row(x2):
        return height - 1 - round((x2 - x2min) / (x2max - x2min) * (height - 1))

    if x1min <= 0.0 <= x1max:  # ось x2
        c = col(0.0)
        for r in range(height):
            grid[r][c] = "|"
    if x2min <= 0.0 <= x2max:    # ось x1
        r = row(0.0)
        for c in range(width):
            grid[r][c] = "-"

    for c in range(width):
        x1 = x1min + (x1max - x1min) * c / (width - 1)
        x2_prev, f_prev = x2min, F([x1, x2min], a)
        for i in range(1, nscan + 1):
            x2 = x2min + (x2max - x2min) * i / nscan
            f_cur = F([x1, x2], a)
            for w, ch in ((0, "#"), (1, "o")):
                if f_prev[w] * f_cur[w] < 0.0:
                    t = f_prev[w] / (f_prev[w] - f_cur[w])
                    x2z = x2_prev + t * (x2 - x2_prev)
                    r = row(x2z)
                    if grid[r][c] in "#o*":
                        if grid[r][c] != ch:
                            grid[r][c] = "*"
                    else:
                        grid[r][c] = ch
            x2_prev, f_prev = x2, f_cur

    print("\nГрафик системы: '#' — кривая F1 = 0, 'o' — кривая F2 = 0, "
          "'*' — пересечения (решения):")
    for r in range(height):
        y = x2max - (x2max - x2min) * r / (height - 1)
        print(f"{y:7.2f} |" + "".join(grid[r]))
    print("       +" + "-" * width)


def column_zeros(x1, w, a, x2r, nscan=300):
    """Нули компоненты w по x2 при фиксированном x1 (скан столбца)."""
    x2min, x2max = x2r
    zeros, x2_prev, f_prev = [], x2min, F([x1, x2min], a)[w]
    for i in range(1, nscan + 1):
        x2 = x2min + (x2max - x2min) * i / nscan
        f_cur = F([x1, x2], a)[w]
        if f_prev * f_cur < 0.0:
            t = f_prev / (f_prev - f_cur)
            zeros.append(x2_prev + t * (x2 - x2_prev))
        x2_prev, f_prev = x2, f_cur
    return zeros


def find_all_solutions(a, x1r, x2r, n1=200, tol=0.15):
    """Поиск пересечений кривых + уточнение Ньютоном. -> [(решение, k_Ньютон)]."""
    x1min, x1max = x1r
    cand = []
    for i in range(n1):
        x1 = x1min + (x1max - x1min) * i / (n1 - 1)
        z1 = column_zeros(x1, 0, a, x2r)
        z2 = column_zeros(x1, 1, a, x2r)
        for u in z1:
            for v in z2:
                if abs(u - v) < tol:
                    cand.append([x1, 0.5 * (u + v)])
    uniq = []
    for c in cand:
        if not any(abs(c[0] - u[0]) < 0.05 and abs(c[1] - u[1]) < 0.05
                   for u in uniq):
            uniq.append(c)
    sols = []
    for c in uniq:
        try:
            s, k = newton(c, a, 1e-6, 100)
        except (RuntimeError, ValueError):
            continue                      # старт из плохой точки — пропускаем
        s = s[-1]
        if (x1min - 1.0 <= s[0] <= x1max + 1.0 and
                x2r[0] - 1.0 <= s[1] <= x2r[1] + 1.0):
            if not any(vec_inf(vec_sub(s, t[0])) < 1e-4 for t in sols):
                sols.append((s, k))
    return sols


# Отчёт

def error_table(xs, x_star, order, header):
    print(header)
    name = "e_k/e_(k-1)" if order == 1 else "e_k/e_(k-1)^2"
    print("    k        x1_k           x2_k         ||x_k - x*||   " + name)
    prev = None
    for k, x in enumerate(xs):
        e = vec_inf(vec_sub(x, x_star))
        if prev is None or prev == 0.0:
            ratio = "        —"
        elif order == 1:
            ratio = f"{e / prev:10.4f}"
        else:
            ratio = f"{e / prev ** 2:10.3e}"
        print(f"  {k:4d}   {x[0]:13.8f}  {x[1]:13.8f}   {e:10.3e}  {ratio}")
        prev = e
    print()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/task2.txt"
    a, x1r, x2r, eps, x0_it, x0_nw = read_task2(path)
    if x1r[0] <= -1.0:
        raise ValueError("левая граница окна по x1 должна быть > -1 (область lg)")

    print("Лабораторная работа 2.2 — вариант 5 (a = %g)" % a)
    print("Система:  x1 - cos(x2) = 1;   x2 - lg(x1 + 1) = %g" % a)
    print(f"Входные данные: {path};  точность eps = {eps:g}")

    # 1) графическое отделение решений
    ascii_plot(a, x1r, x2r)
    print(f"Начальные приближения (по графику, из файла): "
          f"МПИ x0 = ({x0_it[0]:g}, {x0_it[1]:g}), "
          f"Ньютон x0 = ({x0_nw[0]:g}, {x0_nw[1]:g})")

    # 2) все решения и выбор положительного
    sols = find_all_solutions(a, x1r, x2r)
    print("\nВсе решения системы (автопоиск пересечений + уточнение Ньютоном):")
    for s, k in sols:
        tag = "   <-- все компоненты положительны" if all(t > 0 for t in s) else ""
        print("   ({:+.6f}, {:+.6f})   [{k} итер. уточнения]{tag}".format(
            s[0], s[1], k=k, tag=tag))
    pos = [s for s, _ in sols if all(t > 0 for t in s)]
    if not pos:
        raise RuntimeError("Положительное решение не найдено — "
                           "проверьте окно графика")
    x_star = newton(pos[0], a, 1e-12)[0][-1]      # эталон для анализа
    print(f"\nРабочее решение: x* = ({x_star[0]:.9f}, {x_star[1]:.9f})")

    # 3) условие сходимости МПИ
    B = dPhi(x_star, a)
    lams = eig2(B)
    rho = max(abs(z) for z in lams)
    print("\n1) Условие сходимости МПИ:  x1 = 1 + cos(x2),  x2 = a + lg(x1+1)")
    print("   Якобиан Phi'(x*):")
    for rw in B:
        print("      [" + "  ".join(f"{v:+.6f}" for v in rw) + "]")
    print(f"   ||Phi'(x*)||_inf = {matrix_inf_norm(B):.6f}   "
          f"(< 1 — достаточное условие выполнено)")
    for z in lams:
        print(f"   собственное значение Phi':  {z.real:+.6f} {z.imag:+.6f}i")
    print(f"   rho(Phi'(x*)) = {rho:.6f}   "
          f"(< 1 — необходимое и достаточное условие)")
    print("   Значения чисто мнимые -> сходимость колебательная: вектор ошибки")
    print(f"   за два шага умножается на -rho^2 = {-rho * rho:+.4f} "
          f"(по модулю — в {1.0 / rho ** 2:.0f} раза).")

    # 4) МПИ
    print(f"\n2) Метод простой итерации из x0 = ({x0_it[0]:g}, {x0_it[1]:g}):")
    xs_it, k_it = simple_iteration(x0_it, a, eps)
    error_table(xs_it, x_star, 1, f"   Итерации (всего {k_it}):")
    x_fin = xs_it[-1]
    print("   Ответ МПИ: ({:+.9f}, {:+.9f}),  ||F(x)||_inf = {:.2e},  "
          "итераций: {}".format(x_fin[0], x_fin[1], vec_inf(F(x_fin, a)), k_it))
    # константа сжатия на отрезке [x0, x*] - априорная и апостериорная оценки
    q = max(matrix_inf_norm(
        dPhi([x0_it[0] + t * (x_star[0] - x0_it[0]),
              x0_it[1] + t * (x_star[1] - x0_it[1])], a))
        for t in [i / 20.0 for i in range(21)])
    if q < 1.0:
        k_apr = math.ceil(math.log(eps / vec_inf(vec_sub(x0_it, x_star)))
                          / math.log(q))
        ap = q / (1.0 - q) * vec_inf(vec_sub(xs_it[-1], xs_it[-2]))
        print(f"   Априорная оценка (q = {q:.4f} на отрезке [x0, x*]): "
              f"гарантировано не более {k_apr} итераций — пессимистично, "
              f"факт {k_it}")
        print(f"   Апостериорная оценка: ||x_K - x*|| <= q/(1-q)*||dx|| "
              f"<= {ap:.2e}")

    # 5) Ньютон
    Jst = J(x_star, a)
    detJ = Jst[0][0] * Jst[1][1] - Jst[0][1] * Jst[1][0]
    print("\n3) Метод Ньютона:  J(x_k)·Delta_k = -F(x_k),  x_(k+1) = x_k + Delta_k")
    print(f"   det J(x*) = {detJ:.6f} ≠ 0 — линеаризация корректна, "
          f"сходимость квадратичная")
    xs_nw, k_nw = newton(x0_nw, a, eps)
    error_table(xs_nw, x_star, 2, f"   Итерации из x0 = ({x0_nw[0]:g}, "
                                  f"{x0_nw[1]:g}) (всего {k_nw}):")
    x_fin = xs_nw[-1]
    print("   Ответ Ньютона: ({:+.9f}, {:+.9f}),  ||F(x)||_inf = {:.2e},  "
          "итераций: {}".format(x_fin[0], x_fin[1], vec_inf(F(x_fin, a)), k_nw))

    # 6) Анализ
    print("\n4) Анализ зависимости погрешности от числа итераций")
    print("   МПИ:    отношение e_k/e_(k-1) осциллирует (собственные значения")
    print(f"           Phi' мнимые), но за два шага погрешность убывает в "
          f"rho^2 = {rho ** 2:.3f} раза —")
    print(f"           линейная колебательная сходимость; всего {k_it} итераций.")
    print("           Причина: диагональ Phi' нулевая — x1_(k+1) зависит только")
    print("           от x2_k и наоборот, ошибка проходит двухшаговый цикл.")
    print("   Ньютон: отношение e_k/e_(k-1)^2 стабилизируется — квадратичная")
    print(f"           сходимость; всего {k_nw} итераций.")
    print(f"   Итог: Ньютону понадобилось в {k_it / k_nw:.0f} раза меньше итераций.")


if __name__ == "__main__":
    main()