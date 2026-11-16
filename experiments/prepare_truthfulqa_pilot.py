"""
prepare_truthfulqa_pilot.py
---------------------------
Pilot dataset preparation script for the ML Hallucination Detector project.

Workflow:
  1. Loads the 100 reproducibly selected TruthfulQA questions from
     `dataset/truthfulqa_pilot_prompts.csv`.
  2. Loads the Qwen2.5-0.5B-Instruct model and tokenizer ONCE.
  3. Iterates over each prompt and extracts internal signals
     (token probabilities, predictive entropy, generated answer) using
     the verified `src/signal_extraction.py` pipeline.
  4. Preserves all original benchmark metadata (Category, Type, Best Answer,
     Correct Answers, Incorrect Answers, Source URL, Sample ID).
  5. Assigns label = -1 (LABEL_UNLABELED) to every record so no synthetic or
     unverified factuality label is assigned.
  6. Saves the generated pilot records to
     `dataset/truthfulqa_pilot_100_unannotated.csv`.
"""

import os
import sys
import time
import pandas as pd

# Add the project root to sys.path so src imports work cleanly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model_loader import load_model
from src.signal_extraction import extract_signals
from src.dataset_manager import (
    create_dataset_record,
    save_dataset,
    LABEL_UNLABELED,
)


# ---------------------------------------------------------------------------
# Configuration Paths
# ---------------------------------------------------------------------------

PROMPTS_CSV_PATH = os.path.join(PROJECT_ROOT, "dataset", "truthfulqa_pilot_prompts.csv")
OUTPUT_CSV_PATH = os.path.join(PROJECT_ROOT, "dataset", "truthfulqa_pilot_100_unannotated.csv")


def run_pilot(prompts_path: str = PROMPTS_CSV_PATH, output_path: str = OUTPUT_CSV_PATH) -> pd.DataFrame:
    """
    Executes the TruthfulQA pilot generation and signal extraction workflow.

    Args:
        prompts_path (str): Path to truthfulqa_pilot_prompts.csv.
        output_path (str): Destination for truthfulqa_pilot_100_unannotated.csv.

    Returns:
        pd.DataFrame: Completed unannotated pilot dataset.
    """
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"[Error] Prompts file not found at '{prompts_path}'. Run Step 1 first.")

    print("=" * 70)
    print("  ML Hallucination Detector — TruthfulQA Pilot Generation (100 Questions)")
    print("=" * 70)

    # 1. Load prompts
    prompts_df = pd.read_csv(prompts_path, encoding="utf-8")
    total_questions = len(prompts_df)
    print(f"\n[1/3] Loaded {total_questions} prompts from: {prompts_path}")

    # 2. Load model and tokenizer ONCE
    print("\n[2/3] Loading Qwen2.5-0.5B-Instruct model and tokenizer...")
    start_load_time = time.time()
    model, tokenizer = load_model()
    print(f"      Model ready in {time.time() - start_load_time:.2f} seconds.")

    # 3. Process each prompt
    print(f"\n[3/3] Beginning inference and signal extraction across {total_questions} questions...\n")
    records = []
    overall_start_time = time.time()

    for idx, row in prompts_df.iterrows():
        sample_num = idx + 1
        question = row["Question"]
        category = row.get("Category", "Unknown")
        type_str = row.get("Type", "Unknown")
        best_ans = row.get("Best Answer", "")
        correct_ans = row.get("Correct Answers", "")
        incorrect_ans = row.get("Incorrect Answers", "")
        source_url = row.get("Source", "")
        orig_idx = row.get("original_index", idx)
        sample_id = row.get("sample_id", sample_num)

        t0 = time.time()
        # Extract signals using existing verified module
        signals = extract_signals(model, tokenizer, question)
        elapsed = time.time() - t0

        # Construct dataset record with label = -1 (LABEL_UNLABELED)
        record = create_dataset_record(
            question=question,
            generated_answer=signals["answer"],
            label=LABEL_UNLABELED,
            average_probability=signals["average_probability"],
            average_entropy=signals["average_entropy"],
            token_probabilities=signals["token_probabilities"],
            entropy_values=signals["entropy_values"],
            source=f"TruthfulQA ({category})",
            reference_answer=best_ans,
            allow_unlabeled=True,
            # Benchmark metadata preserved
            sample_id=sample_id,
            original_index=orig_idx,
            category=category,
            type=type_str,
            best_answer=best_ans,
            correct_answers=correct_ans,
            incorrect_answers=incorrect_ans,
            source_url=source_url,
        )
        records.append(record)

        print(
            f"[{sample_num:03d}/{total_questions:03d}] ({category}) "
            f"Tokens: {len(signals['token_probabilities']):<3} | "
            f"AvgProb: {signals['average_probability']:.4f} | "
            f"AvgEntropy: {signals['average_entropy']:.4f} nats | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpoint save every 10 questions to ensure safe persistence
        if sample_num % 10 == 0 or sample_num == total_questions:
            save_dataset(records, output_path, allow_unlabeled=True)

    total_time = time.time() - overall_start_time
    print("\n" + "=" * 70)
    print(f"  PILOT GENERATION COMPLETE in {total_time:.2f}s ({total_time / 60:.2f} min)")
    print(f"  Average time per question: {total_time / total_questions:.2f}s")
    print(f"  Saved {len(records)} unannotated records to: {output_path}")
    print("=" * 70)

    return pd.DataFrame(records)


if __name__ == "__main__":
    run_pilot()
