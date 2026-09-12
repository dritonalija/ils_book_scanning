#!/usr/bin/env python3
"""Build the structural catalog of every benchmark instance.

Reads the instance files listed in ``instances.txt`` and ``instances-test.txt``
and writes one row per instance: size, totals, and the averages that describe
the shape of the instance. The catalog is the input for instance-level
summaries and for the scale plots.

Everything this script reads is tracked in the repository, so a fresh clone can
run it.

    python build_instance_catalog.py
    python build_instance_catalog.py --output experiments/instance_catalog.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_LISTS = ["instances.txt", "instances-test.txt"]
DEFAULT_OUTPUT = "experiments/instance_catalog.csv"

COLUMNS = [
    "dataset",
    "instance_name",
    "books",
    "libraries",
    "days",
    "book_occurrences",
    "available_books",
    "available_score",
    "avg_book_value",
    "avg_frequency",
    "avg_signup",
    "avg_rate",
    "max_books_in_library",
]


def read_instance_list(path: str) -> list[str]:
    """Return the instance paths in a list file, ignoring blanks and comments."""
    values: list[str] = []
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        item = line.strip().replace("\\", "/")
        if item and not item.startswith("#"):
            values.append(item)
    return values


def instance_name(value: str) -> str:
    return Path(str(value).replace("\\", "/")).name


def dataset_name(path: str) -> str:
    """Return the dataset folder of an ``input/<dataset>/<file>`` path."""
    parts = str(path).replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[0] == "input":
        return parts[1]
    return "unknown"


def parse_instance(path: str) -> dict[str, object]:
    """Read one instance file and measure its structure.

    The file format is the Hash Code 2020 one: a header line with the book,
    library and day counts, a line of book scores, then two lines per library.
    """
    full = ROOT / path
    with full.open("r", encoding="utf-8") as handle:
        first = handle.readline().strip().split()
        num_books, num_libs, num_days = map(int, first)
        scores = list(map(int, handle.readline().strip().split()))
        occurrences = 0
        signup_sum = 0
        rate_sum = 0
        books_available: set[int] = set()
        max_books_in_lib = 0
        for _ in range(num_libs):
            n_books, signup, rate = map(int, handle.readline().strip().split())
            ids = list(map(int, handle.readline().strip().split()))
            occurrences += n_books
            signup_sum += signup
            rate_sum += rate
            max_books_in_lib = max(max_books_in_lib, n_books)
            books_available.update(ids)
    available_score = sum(scores[book_id] for book_id in books_available)
    avg_freq = occurrences / max(1, len(books_available))
    return {
        "dataset": dataset_name(path),
        "instance": path,
        "instance_name": instance_name(path),
        "books": num_books,
        "libraries": num_libs,
        "days": num_days,
        "book_occurrences": occurrences,
        "available_books": len(books_available),
        "available_score": available_score,
        "avg_book_value": sum(scores) / max(1, num_books),
        "avg_frequency": avg_freq,
        "avg_signup": signup_sum / max(1, num_libs),
        "avg_rate": rate_sum / max(1, num_libs),
        "max_books_in_library": max_books_in_lib,
    }


def build_catalog(lists: list[str]) -> pd.DataFrame:
    instances: list[str] = []
    for name in lists:
        instances.extend(read_instance_list(name))
    catalog = pd.DataFrame(parse_instance(path) for path in instances)
    catalog = catalog.sort_values(["dataset", "instance_name"]).reset_index(drop=True)
    return catalog[COLUMNS]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure every benchmark instance and write the catalog."
    )
    parser.add_argument(
        "--instance-list",
        nargs="+",
        default=DEFAULT_LISTS,
        help="List files naming the instances to measure.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Destination CSV, relative to the repository root.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    catalog = build_catalog(args.instance_list)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output, index=False)
    print(f"{len(catalog)} instances -> {args.output}")


if __name__ == "__main__":
    main()
