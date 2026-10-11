"""Data loader and pilot sample generator for A-OKVQA Visual Question Answering.

Loads A-OKVQA validation split via Hugging Face `datasets` (or parquet fallback),
samples N=200 questions with a fixed seed, and saves pilot item IDs to results/derived/pilot_ids.json.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

# Dataset identifier and revision tracking
AOKVQA_DATASET_ID = "HuggingFaceM4/A-OKVQA"
AOKVQA_REVISION = "d1b0efa3a436e9101dfbde3752db7607da696c35"


def get_dataset_metadata() -> Dict[str, str]:
    """Return dataset identifier and revision used for reproducibility."""
    return {
        "dataset_id": AOKVQA_DATASET_ID,
        "revision": AOKVQA_REVISION,
        "split": "validation",
    }


def sample_pilot_indices(total_items: int, n: int = 200, seed: int = 0) -> List[int]:
    """Sample n sorted unique indices with a fixed seed.

    Parameters
    ----------
    total_items : int
        Total number of items in the split.
    n : int, default=200
        Number of items to sample.
    seed : int, default=0
        RNG seed.

    Returns
    -------
    List[int]
        Sorted list of sampled indices.
    """
    if n > total_items:
        raise ValueError(f"Requested sample size {n} exceeds total items {total_items}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(total_items, size=n, replace=False)
    chosen.sort()
    return [int(x) for x in chosen]


def load_aokvqa_validation(
    pilot_ids_path: Optional[str] = "results/derived/pilot_ids.json",
    n_sample: int = 200,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Load A-OKVQA validation multiple choice examples, sampling N=200 items.

    Tries `datasets.load_dataset` first; if `datasets` is not installed, falls back
    to pandas parquet reader.

    Parameters
    ----------
    pilot_ids_path : Optional[str]
        Path to save/load sampled pilot question IDs (write once).
    n_sample : int, default=200
        Number of items to sample if pilot_ids_path does not exist.
    seed : int, default=0
        RNG seed for pilot sampling.

    Returns
    -------
    List[Dict[str, Any]]
        List of dicts: {'question_id', 'question', 'choices', 'correct_choice_idx', 'image'}.
    """
    meta = get_dataset_metadata()
    print(f"Loading A-OKVQA dataset: {meta['dataset_id']} (revision: {meta['revision']})")

    # 1. Load full validation set
    raw_examples = []
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset(meta["dataset_id"], split="validation", revision=meta["revision"])
        for item in ds:
            raw_examples.append(item)
    except (ImportError, Exception) as exc:
        # Fallback to pandas parquet
        import io
        import pandas as pd
        from PIL import Image

        parquet_url = "https://huggingface.co/api/datasets/HuggingFaceM4/A-OKVQA/parquet/default/validation/0.parquet"
        df = pd.read_parquet(parquet_url)
        for _, row in df.iterrows():
            img_data = row["image"]
            if isinstance(img_data, dict) and "bytes" in img_data:
                img = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
            else:
                img = img_data
            raw_examples.append({
                "question_id": str(row["question_id"]),
                "question": str(row["question"]),
                "choices": list(row["choices"]),
                "correct_choice_idx": int(row["correct_choice_idx"]),
                "image": img,
            })

    total_len = len(raw_examples)
    q_to_example = {ex["question_id"]: ex for ex in raw_examples}

    # 2. Check if pilot_ids_path exists; if not, sample and write once
    target_path = Path(pilot_ids_path) if pilot_ids_path else None
    if target_path and target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            saved_meta = json.load(f)
        saved_qids = saved_meta["question_ids"]
        print(f"Loaded {len(saved_qids)} pilot item IDs from {target_path}")
        return [q_to_example[qid] for qid in saved_qids if qid in q_to_example]

    # Sample deterministically
    sampled_indices = sample_pilot_indices(total_len, n=n_sample, seed=seed)
    sampled_examples = [raw_examples[i] for i in sampled_indices]
    sampled_qids = [ex["question_id"] for ex in sampled_examples]

    if target_path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        from src.io_utils import write_raw
        record = {
            "dataset_id": meta["dataset_id"],
            "revision": meta["revision"],
            "split": meta["split"],
            "n_sample": len(sampled_qids),
            "seed": seed,
            "question_ids": sampled_qids,
        }
        write_raw(target_path, record)
        print(f"Sampled N={len(sampled_qids)} items and wrote pilot IDs once to {target_path}")

    return sampled_examples


def sample_main_disjoint_ids(
    pilot_ids_path: str = "results/derived/pilot_ids.json",
    main_ids_path: str = "results/derived/main_ids.json",
    n_sample: int = 500,
    seed: int = 1,
) -> List[str]:
    """Sample N=500 validation question IDs strictly disjoint from pilot_ids.json.

    Writes results write-once to main_ids_path via write_raw.

    Parameters
    ----------
    pilot_ids_path : str, default="results/derived/pilot_ids.json"
        Path to pilot IDs file.
    main_ids_path : str, default="results/derived/main_ids.json"
        Destination path for main study question IDs.
    n_sample : int, default=500
        Number of items to sample.
    seed : int, default=1
        RNG seed for disjoint sampling.

    Returns
    -------
    List[str]
        Sampled question IDs for the main study.
    """
    target_path = Path(main_ids_path)
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(data["question_ids"])

    # Load pilot IDs to exclude
    with open(pilot_ids_path, "r", encoding="utf-8") as f:
        pilot_data = json.load(f)
    pilot_ids_set = set(pilot_data["question_ids"])

    meta = get_dataset_metadata()
    # Fetch all validation question IDs
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset(meta["dataset_id"], split="validation", revision=meta["revision"])
        all_qids = [str(x["question_id"]) for x in ds]
    except Exception:
        import pandas as pd
        parquet_url = "https://huggingface.co/api/datasets/HuggingFaceM4/A-OKVQA/parquet/default/validation/0.parquet"
        df = pd.read_parquet(parquet_url, columns=["question_id"])
        all_qids = [str(x) for x in df["question_id"]]

    available_qids = [qid for qid in all_qids if qid not in pilot_ids_set]
    if n_sample > len(available_qids):
        raise ValueError(
            f"Requested {n_sample} items, but only {len(available_qids)} disjoint items remain."
        )

    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(len(available_qids), size=n_sample, replace=False)
    chosen_indices.sort()
    main_qids = [available_qids[i] for i in chosen_indices]

    from src.io_utils import write_raw
    record = {
        "dataset_id": meta["dataset_id"],
        "revision": meta["revision"],
        "split": meta["split"],
        "n_sample": len(main_qids),
        "seed": seed,
        "pilot_ids_path": pilot_ids_path,
        "question_ids": main_qids,
    }
    write_raw(target_path, record)
    print(f"Sampled N={len(main_qids)} disjoint items and wrote to {target_path}")
    return main_qids


if __name__ == "__main__":
    meta = get_dataset_metadata()
    print(f"Dataset ID: {meta['dataset_id']}")
    print(f"Revision: {meta['revision']}")
    items = load_aokvqa_validation()
    print(f"Loaded {len(items)} pilot items.")
    main_qids = sample_main_disjoint_ids()
    print(f"Loaded {len(main_qids)} main items.")

