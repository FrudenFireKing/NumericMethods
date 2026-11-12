"""Запуск всех частей лабораторной №3 по порядку."""
import subprocess
import sys

SCRIPTS = ["task1_interp.py", "task2_spline.py", "task3_lsm.py",
           "task4_diff.py", "task5_integrate.py"]

for script in SCRIPTS:
    print("\n" + "=" * 72 + f"\n>>> {script}\n" + "=" * 72 + "\n")
    if subprocess.run([sys.executable, script]).returncode != 0:
        sys.exit(f"Ошибка в {script}")