# Запуск всех частей лабораторной по порядку
import subprocess
import sys

SCRIPTS = ["task1_lu.py", "task2_tridiagonal.py", "task3_iterations.py",
           "task4_jacobi.py", "task5_qr.py"]

for script in SCRIPTS:
    print("\n" + "=" * 72 + f"\n>>> {script}\n" + "=" * 72 + "\n")
    if subprocess.run([sys.executable, script]).returncode != 0:
        sys.exit(f"Ошибка в {script}")