"""Запуск всех частей лабораторной №2 по порядку."""
import subprocess
import sys

SCRIPTS = ["task1_nonlinear.py", "task2_systems.py"]

for script in SCRIPTS:
    print("\n" + "=" * 72 + f"\n>>> {script}\n" + "=" * 72 + "\n")
    if subprocess.run([sys.executable, script]).returncode != 0:
        sys.exit(f"Ошибка в {script}")