"""Small command-line wrapper for the notebook and saved SQLite reports."""

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Run churn analysis or show SQL reports")
    parser.add_argument("command", choices=["analyze", "sql"])
    args = parser.parse_args()

    if args.command == "analyze":
        subprocess.run(
            [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
             "--inplace", "churn_analysis.ipynb", "--ExecutePreprocessor.kernel_name=customer-churn"],
            cwd=ROOT,
            check=True,
        )
        return

    database = ROOT / "data" / "churn.sqlite"
    if not database.exists():
        parser.error("data/churn.sqlite is missing; run 'analyze' first")
    statements = [item.strip() for item in (ROOT / "analysis.sql").read_text(encoding="utf-8").split(";")
                  if item.strip()]
    with sqlite3.connect(database) as connection:
        for number, statement in enumerate(statements, 1):
            cursor = connection.execute(statement)
            print(f"\nQuery {number}")
            print(" | ".join(column[0] for column in cursor.description))
            for row in cursor.fetchall():
                print(" | ".join(str(value) for value in row))


if __name__ == "__main__":
    main()
