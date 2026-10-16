"""
Лабораторная работа 2.1 — методы простой итерации и Ньютона
для нелинейного уравнения.  ВАРИАНТ 5:

    f(x) = cos(x) + 0.25*x - 0.5 = 0

Схема решения:
  1) отделение корней: ASCII-график f(x) + сканирование смен знака
     (начальное приближение x0 выбирается по графику, задаётся в файле);
  2) уточнение положительного корня двумя методами:
       МПИ    — эквивалентная форма x = arccos(0.5 - 0.25x),
                |phi'(x)| ~ 0.25 < 1 — сходимость гарантирована;
                (контрпример: форма x = 2 - 4cos x, |phi'| ~ 4 — расходится);
       Ньютон — x_{k+1} = x_k - f(x_k)/f'(x_k), квадратичная сходимость;
  3) анализ зависимости погрешности от числа итераций: таблицы |x_k - x*|
     и отношений e_k/e_{k-1} (-> q, линейная) и e_k/e_{k-1}^2 (-> M, квадратичная).

Эталонный корень x* для анализа погрешности считается бисекцией
(гарантированно сходящийся метод) с высокой точностью.

Вход: data/task21.txt — окно графика; eps; x0_МПИ; x0_Ньютон.
Запуск: python task1_nonlinear.py [файл]
"""

import math
import sys

from common import read_task1

MAX_ITER = 1000

def f(x):
    return math.cos(x) + 0.25 * x - 0.5


def df(x):
    return -math.sin(x) + 0.25


def d2f(x):
    return -math.cos(x)


def phi(x):
    #Сходящаяся эквивалентная форма:  x = arccos(0.5 - 0.25*x)
    return math.acos(0.5 - 0.25 * x)


def dphi(x):
    return 0.25 / math.sqrt(1.0 - (0.5 - 0.25 * x) ** 2)


def phi_bad(x):
    #НЕудачная форма (демонстрация расходимости):  x = 2 - 4*cos x
    return 2.0 - 4.0 * math.cos(x)


# график, сканирование, бисекция

def ascii_plot(xmin, xmax, width=72, height=17):
    #ASCII-график f(x): '*' — кривая, '-' — ось y = 0, '|' — ось x = 0
    xs = [xmin + (xmax - xmin) * i / (width - 1) for i in range(width)]
    ys = [f(x) for x in xs]
    ymin, ymax = min(ys), max(ys)
    if ymax - ymin < 1e-12:
        ymin, ymax = ymin - 1.0, ymax + 1.0

    def row(y):
        return height - 1 - round((y - ymin) / (ymax - ymin) * (height - 1))

    grid = [[" "] * width for _ in range(height)]
    if ymin <= 0.0 <= ymax:                       # ось абсцисс
        r = row(0.0)
        for c in range(width):
            grid[r][c] = "-"
    if xmin <= 0.0 <= xmax:                       # ось ординат
        c = round((0.0 - xmin) / (xmax - xmin) * (width - 1))
        for r in range(height):
            if grid[r][c] == " ":
                grid[r][c] = "|"
    for c, y in enumerate(ys):                    # кривая
        grid[row(y)][c] = "*"

    print(f"\nГрафик f(x) на [{xmin:g}, {xmax:g}] "
          f"(пересечения кривой с линией '-' — корни):")
    for r in range(height):
        y = ymax - (ymax - ymin) * r / (height - 1)
        print(f"{y:8.2f} |" + "".join(grid[r]))
    print("         +" + "-" * width)


def scan_brackets(xmin, xmax, n=4000):
    #Отрезки со сменой знака f — по одному корню в каждом
    out, dx = [], (xmax - xmin) / n
    x_prev, f_prev = xmin, f(xmin)
    for i in range(1, n + 1):
        x = xmin + i * dx
        fx = f(x)
        if f_prev * fx < 0.0:
            out.append((x_prev, x))
        x_prev, f_prev = x, fx
    return out


def bisection(lo, hi, tol=1e-14, max_iter=200):
    #Эталонное уточнение корня на отрезке смены знака
    f_lo = f(lo)
    for _ in range(max_iter):
        m = 0.5 * (lo + hi)
        fm = f(m)
        if fm == 0.0 or (hi - lo) < 2.0 * tol:
            return m
        if f_lo * fm < 0.0:
            hi = m
        else:
            lo, f_lo = m, fm
    return 0.5 * (lo + hi)


# методы

def simple_iteration(x0, eps, max_iter=MAX_ITER):
    #МПИ: x_{k+1} = phi(x_k); останов |x_{k+1} - x_k| < eps
    xs = [x0]
    for k in range(1, max_iter + 1):
        try:
            x = phi(xs[-1])
        except ValueError:
            raise RuntimeError("итерация вышла из области определения phi") from None
        xs.append(x)
        if abs(x - xs[-2]) < eps:
            return xs, k
    raise RuntimeError(f"МПИ не сошёлся за {max_iter} итераций — "
                       f"проверьте условие |phi'| < 1")


def newton(x0, eps, max_iter=100):
    # Метод Ньютона: x_{k+1} = x_k - f(x_k)/f'(x_k)
    xs = [x0]
    for k in range(1, max_iter + 1):
        xk = xs[-1]
        d = df(xk)
        if abs(d) < 1e-14:
            raise ZeroDivisionError("f'(x) ~ 0 — метод Ньютона неприменим")
        x = xk - f(xk) / d
        xs.append(x)
        if abs(x - xk) < eps:
            return xs, k
    raise RuntimeError(f"Метод Ньютона не сошёлся за {max_iter} итераций")


def error_table(xs, x_star, order, header):
    # Таблица погрешностей; order = 1 -> e_k/e_(k-1), order = 2 -> e_k/e_(k-1)^2
    print(header)
    name = "e_k/e_(k-1)" if order == 1 else "e_k/e_(k-1)^2"
    print("    k          x_k             |x_k - x*|   " + name)
    prev = None
    for k, x in enumerate(xs):
        e = abs(x - x_star)
        if prev is None or prev == 0.0:
            ratio = "        —"
        elif order == 1:
            ratio = f"{e / prev:10.6f}"
        else:
            ratio = f"{e / prev ** 2:10.3e}"
        print(f"  {k:4d}   {x:16.10f}   {e:10.3e}  {ratio}")
        prev = e
    print()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/task1.txt"
    xmin, xmax, eps, x0_iter, x0_newton = read_task1(path)

    print("Лабораторная работа 2.1 — вариант 5")
    print("Уравнение: cos(x) + 0.25*x - 0.5 = 0")
    print(f"Входные данные: {path};  точность eps = {eps:g}")

    # 1) отделение корней: график + скан
    ascii_plot(xmin, xmax)
    roots = [bisection(lo, hi) for lo, hi in scan_brackets(xmin, xmax)]
    print("\nОтделение корней (сканирование + бисекция):")
    for r in roots:
        tag = "   <-- положительный" if r > 0 else ""
        print(f"   x* ~ {r:+.9f}{tag}")
    pos = [r for r in roots if r > 0]
    if not pos:
        raise RuntimeError("В окне графика нет положительных корней — "
                           "измените окно в data/task21.txt")
    target = min(pos, key=lambda r: abs(r - x0_iter))
    print(f"\nРабочий положительный корень: x* = {target:+.9f} "
          f"(ближайший к x0 = {x0_iter:g}, взятому по графику)")
    if len(pos) > 1:
        others = ", ".join(f"{r:.6f}" for r in pos if r != target)
        print(f"Замечание: есть ещё положительный корень {others} — Ньютон найдёт "
              f"его из x0 ~ 4 (достаточно поменять x0 в файле).")

    # 2) метод простой итерации
    lo, hi = sorted((x0_iter, target))
    q = max(dphi(lo + (hi - lo) * i / 20) for i in range(21))
    print("\n1) Метод простой итерации:  x = arccos(0.5 - 0.25*x)")
    print(f"   На отрезке [{lo:g}, {hi:g}]:  |phi'(x)| <= q = {q:.6f} < 1 — "
          f"сходимость гарантирована (сжимающее отображение)")
    xs_it, k_it = simple_iteration(x0_iter, eps)
    error_table(xs_it, target, 1, f"   Итерации МПИ из x0 = {x0_iter:g} "
                                  f"(всего {k_it}):")
    x_mpi = xs_it[-1]
    ap = q / (1.0 - q) * abs(xs_it[-1] - xs_it[-2])
    print(f"   Ответ МПИ: x = {x_mpi:+.10f},  |f(x)| = {abs(f(x_mpi)):.2e}, "
          f"итераций: {k_it}")
    print(f"   Апостериорная оценка: |x_K - x*| <= q/(1-q)*|x_K - x_(K-1)| "
          f"<= {ap:.2e}  (факт: {abs(x_mpi - target):.2e})")

    # --- 3) метод Ньютона ---
    M = abs(d2f(target) / (2.0 * df(target)))
    print("\n2) Метод Ньютона:  x_(k+1) = x_k - f(x_k)/f'(x_k)")
    print(f"   Теория: e_(k+1) ~ |f''/(2f')|*e_k^2, константа M = {M:.6f}")
    xs_nw, k_nw = newton(x0_newton, eps)
    error_table(xs_nw, target, 2, f"   Итерации Ньютона из x0 = {x0_newton:g} "
                                  f"(всего {k_nw}):")
    x_nwt = xs_nw[-1]
    print(f"   Ответ Ньютона: x = {x_nwt:+.10f},  |f(x)| = {abs(f(x_nwt)):.2e}, "
          f"итераций: {k_nw}")

    # 4) контрпример: неудачная эквивалентная форма
    print("\n3) Контроль: неудачная форма  x = 2 - 4*cos x")
    q_bad = max(4.0 * abs(math.sin(lo + (hi - lo) * i / 20)) for i in range(21))
    print(f"   На том же отрезке |phi'(x)| = 4*|sin x| >= {q_bad:.3f} > 1 — "
          f"сходимости нет; итерации блуждают:")
    x, vals = x0_iter, []
    for k in range(1, 9):
        x = phi_bad(x)
        vals.append(f"x_{k} = {x:+.4f}")
    print("      " + ", ".join(vals))

    # 5) вывод
    print("\n4) Анализ зависимости погрешности от числа итераций")
    print(f"   МПИ:    e_k/e_(k-1) -> q = {q:.4f} — линейная сходимость "
          f"(~{-math.log10(q):.1f} верного знака за итерацию); {k_it} итераций.")
    print(f"   Ньютон: e_k/e_(k-1)^2 -> M = {M:.4f} — квадратичная сходимость "
          f"(число верных знаков удваивается); {k_nw} итер.")


if __name__ == "__main__":
    main()