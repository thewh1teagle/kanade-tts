import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from tqdm import tqdm
from kanade_tokenizer import KanadeModel, load_audio

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from tokenization import load_tokenizer

ROOT = Path(__file__).parent.parent
DATASET = ROOT / "dataset"
METADATA = DATASET / "metadata.csv"
OUTPUT = DATASET / "dataset.jsonl"
MODEL_NAME = "frothywater/kanade-25hz-clean"
NUM_WORKERS = 4


def encode_row(row, model, tokenizer, device):
    file_id = row["file_id"]
    ipa = row["ipa"]

    audio = load_audio(DATASET / file_id, sample_rate=model.config.sample_rate).to(device)

    with torch.inference_mode():
        features = model.encode(audio)

    return {
        "file_id": file_id,
        "ipa": ipa,
        "ipa_tokens": tokenizer.encode(ipa, add_special_tokens=True),
        "audio_tokens": features.content_token_indices.cpu().tolist(),
        "speaker_embedding": features.global_embedding.cpu().tolist(),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Kanade model {MODEL_NAME}...")
    model = KanadeModel.from_pretrained(MODEL_NAME).eval().to(device)

    print("Loading IPA tokenizer...")
    tokenizer = load_tokenizer()

    with open(METADATA) as f:
        rows = list(csv.DictReader(f, delimiter="|"))

    # Load already-processed file_ids
    existing = set()
    if OUTPUT.exists():
        with open(OUTPUT) as f:
            for line in f:
                try:
                    existing.add(json.loads(line)["file_id"])
                except Exception:
                    pass
        print(f"Skipping {len(existing)} already processed records")

    rows_to_process = [(i, row) for i, row in enumerate(rows) if row["file_id"] not in existing]

    with open(OUTPUT, "a") as out:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(encode_row, row, model, tokenizer, device): i for i, row in rows_to_process}
            pending = {}
            next_idx = 0
            for future in tqdm(as_completed(futures), total=len(rows_to_process), desc="Encoding"):
                idx = futures[future]
                pending[idx] = future.result()
                while next_idx in pending:
                    out.write(json.dumps(pending.pop(next_idx), ensure_ascii=False) + "\n")
                    out.flush()
                    next_idx += 1

    print(f"Done. Processed {len(rows_to_process)} new records, {len(existing)} skipped")


if __name__ == "__main__":
    main()
