import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATASET = ROOT / "dataset"
OUTPUT = DATASET / "metadata.csv"


def iter_hebrew():
    with open(DATASET / "he/metadata.csv") as f:
        for row in csv.reader(f, delimiter="|"):
            index, ipa = row[0], row[1]
            yield f"he/wav/{index}.wav", ipa



def iter_english():
    with open(DATASET / "en/metadata.csv") as f:
        for row in csv.reader(f, delimiter="|"):
            filename, ipa = row[0], row[1]
            yield f"en/wav/{filename}", ipa


def main():
    rows = list(iter_hebrew()) + list(iter_english())
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["file_id", "ipa"])
        writer.writerows(rows)
    print(f"Written {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
