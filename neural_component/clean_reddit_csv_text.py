import argparse
import csv
import re
import unicodedata
from pathlib import Path


DEFAULT_INPUT_CSV = "reddit_flat_earth_source_posts.csv"
DEFAULT_OUTPUT_CSV = "reddit_flat_earth_source_posts_clean.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean weird characters and common encoding artifacts from a Reddit CSV "
            "while preserving the CSV structure."
        )
    )
    parser.add_argument(
        "--input-csv",
        default=DEFAULT_INPUT_CSV,
        help=f"CSV file to clean. Default: {DEFAULT_INPUT_CSV}",
    )
    parser.add_argument(
        "--output-csv",
        default=DEFAULT_OUTPUT_CSV,
        help=f"Where to write the cleaned CSV. Default: {DEFAULT_OUTPUT_CSV}",
    )
    parser.add_argument(
        "--ascii-only",
        action="store_true",
        help="Convert cleaned text to ASCII only by dropping remaining non-ASCII characters.",
    )
    return parser.parse_args()


def repair_mojibake(value: str) -> str:
    if not value:
        return value

    suspicious_markers = ("â", "Ã", "ð", "œ", "ž", "\ufffd")
    if any(marker in value for marker in suspicious_markers):
        try:
            repaired = value.encode("latin-1").decode("utf-8")
            return repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            return value
    return value


def strip_control_chars(value: str) -> str:
    return "".join(
        char
        for char in value
        if char in ("\n", "\r", "\t") or unicodedata.category(char)[0] != "C"
    )


def normalize_text(value: str, ascii_only: bool) -> str:
    cleaned = repair_mojibake(value)
    cleaned = unicodedata.normalize("NFKC", cleaned)

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\xa0": " ",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)

    cleaned = strip_control_chars(cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\r\n?", "\n", cleaned)

    if ascii_only:
        cleaned = cleaned.encode("ascii", "ignore").decode("ascii")

    return cleaned.strip()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    with input_path.open("r", newline="", encoding="utf-8", errors="replace") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"No header row found in {input_path}.")

        cleaned_rows = []
        for row in reader:
            cleaned_rows.append(
                {
                    key: normalize_text(value or "", ascii_only=args.ascii_only)
                    for key, value in row.items()
                }
            )

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    print(f"Cleaned CSV written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
