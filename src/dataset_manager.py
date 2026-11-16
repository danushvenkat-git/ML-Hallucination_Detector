"""
dataset_manager.py
------------------
Dataset management module for the ML Hallucination Detector project.

Handles creating, validating, saving, loading, and deduplicating labelled
dataset records containing internal LLM signals (token probabilities,
predictive entropy) alongside ground-truth labels (0 = factual, 1 = hallucinated).

This is Phase 3 of the ML Hallucination Detector project.
No classifier model is trained here.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
import pandas as pd


# ---------------------------------------------------------------------------
# Constants & Schema Definition
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "question",
    "generated_answer",
    "label",
    "average_probability",
    "average_entropy",
    "token_probabilities",
    "entropy_values",
    "source",
    "reference_answer",
]

# Ground-truth labels:
#   -1 = Unlabeled / Awaiting annotation
#    0 = Factual / Accurate
#    1 = Hallucinated / Inaccurate
LABEL_UNLABELED = -1
LABEL_FACTUAL = 0
LABEL_HALLUCINATED = 1

VALID_LABELS = {LABEL_FACTUAL, LABEL_HALLUCINATED}           # Strictly {0, 1}
ALL_ALLOWED_LABELS = {LABEL_UNLABELED, LABEL_FACTUAL, LABEL_HALLUCINATED}  # {-1, 0, 1}

# Columns that must never contain null / NaN values
NON_NULL_COLUMNS = [
    "question",
    "generated_answer",
    "label",
    "average_probability",
    "average_entropy",
]


# ---------------------------------------------------------------------------
# Helper: Record Creation
# ---------------------------------------------------------------------------

def create_dataset_record(
    question: str,
    generated_answer: str,
    label: int,
    average_probability: float,
    average_entropy: float,
    token_probabilities: List[float],
    entropy_values: List[float],
    source: str = "manual",
    reference_answer: str = "",
    allow_unlabeled: bool = False,
    **extra_metadata: Any,
) -> Dict[str, Any]:
    """
    Creates a single structured dataset record dictionary.

    Args:
        question (str): Prompt or question sent to the LLM.
        generated_answer (str): Answer text output by the LLM.
        label (int): Ground truth (0 = factual, 1 = hallucinated, or -1 = unannotated).
        average_probability (float): Mean probability across generated tokens.
        average_entropy (float): Mean predictive entropy across generated tokens.
        token_probabilities (list[float]): Per-token probability values.
        entropy_values (list[float]): Per-token predictive entropy values.
        source (str): Origin benchmark/dataset (e.g., 'manual', 'truthful_qa').
        reference_answer (str): Ground-truth reference answer if available.
        allow_unlabeled (bool): If True, allows label=-1 (unannotated). Default False.
        **extra_metadata: Optional additional benchmark fields to preserve (e.g., category, type, sample_id).

    Returns:
        dict: Standardized dataset record matching the schema plus extra metadata.

    Raises:
        ValueError: If label is invalid or fields are improperly typed.
    """
    valid_set = ALL_ALLOWED_LABELS if allow_unlabeled else VALID_LABELS

    if label not in valid_set:
        allowed_desc = "{0, 1}" if not allow_unlabeled else "{-1, 0, 1}"
        raise ValueError(
            f"[Error] Invalid label: {label}. Allowed labels under allow_unlabeled={allow_unlabeled} are {allowed_desc}."
        )

    record = {
        "question": str(question).strip(),
        "generated_answer": str(generated_answer).strip(),
        "label": int(label),
        "average_probability": float(average_probability),
        "average_entropy": float(average_entropy),
        "token_probabilities": list(token_probabilities),
        "entropy_values": list(entropy_values),
        "source": str(source).strip(),
        "reference_answer": str(reference_answer).strip(),
    }
    # Preserve any extra metadata fields (e.g., category, type, sample_id, etc.)
    record.update(extra_metadata)
    return record


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_dataframe(df: pd.DataFrame, allow_unlabeled: bool = False) -> None:
    """
    Validates that a DataFrame conforms strictly to the dataset schema.

    Checks:
        1. All required columns are present.
        2. Non-null columns contain no missing/NaN values.
        3. Labels are strictly in {0, 1} (or {-1, 0, 1} if allow_unlabeled=True).
        4. Probability values are bounded in [0.0, 1.0].
        5. Entropy values are non-negative (>= 0.0).

    Args:
        df (pd.DataFrame): Dataset to validate.
        allow_unlabeled (bool): If True, accepts label=-1 for unannotated data.

    Raises:
        ValueError: If any validation rule is violated.
    """
    if df.empty:
        raise ValueError("[Error] DataFrame is empty. Cannot validate an empty dataset.")

    # 1. Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"[Error] Dataset is missing required column(s): {missing_cols}. "
            f"Expected schema: {REQUIRED_COLUMNS}"
        )

    # 2. Check for missing values in essential fields
    for col in NON_NULL_COLUMNS:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise ValueError(
                f"[Error] Column '{col}' contains {null_count} null/missing value(s). "
                f"Essential fields must not be empty."
            )

    # 3. Check labels
    valid_set = ALL_ALLOWED_LABELS if allow_unlabeled else VALID_LABELS
    unique_labels = set(df["label"].unique())
    invalid_labels = unique_labels - valid_set
    if invalid_labels:
        allowed_desc = "{0, 1}" if not allow_unlabeled else "{-1 (unannotated), 0 (factual), 1 (hallucinated)}"
        raise ValueError(
            f"[Error] Found invalid label(s): {invalid_labels}. "
            f"Allowed labels under allow_unlabeled={allow_unlabeled} are strictly {allowed_desc}."
        )

    # 4. Check probability bounds
    prob_min = df["average_probability"].min()
    prob_max = df["average_probability"].max()
    if prob_min < 0.0 or prob_max > 1.0:
        raise ValueError(
            f"[Error] 'average_probability' values out of valid range [0.0, 1.0]: min={prob_min}, max={prob_max}."
        )

    # 5. Check entropy bounds (entropy is non-negative)
    ent_min = df["average_entropy"].min()
    if ent_min < 0.0:
        raise ValueError(
            f"[Error] 'average_entropy' cannot be negative: min={ent_min}."
        )


def get_labeled_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts only strictly labeled records (0 = factual, 1 = hallucinated)
    for machine learning training and evaluation.

    Guarantees that no unannotated (-1) records leak into classifier training.

    Args:
        df (pd.DataFrame): Dataset DataFrame.

    Returns:
        pd.DataFrame: Clean subset with only labels in {0, 1}.

    Raises:
        ValueError: If the resulting labeled dataset is empty.
    """
    validate_dataframe(df, allow_unlabeled=True)
    labeled_df = df[df["label"].isin(VALID_LABELS)].copy()
    if labeled_df.empty:
        raise ValueError(
            "[Error] No labeled records (label 0 or 1) found. Dataset contains only unannotated records (-1)."
        )
    return labeled_df


# ---------------------------------------------------------------------------
# List Serialization & Deserialization
# ---------------------------------------------------------------------------

def _serialize_list_field(val: Any) -> str:
    """Converts a Python list to a JSON string for safe CSV storage."""
    if isinstance(val, (list, tuple)):
        return json.dumps(val)
    elif isinstance(val, str):
        # Already a string (e.g., from prior serialization)
        return val
    else:
        return json.dumps([])


def _deserialize_list_field(val: Any) -> List[float]:
    """Safely parses a JSON string back into a Python list of floats."""
    if isinstance(val, list):
        return [float(x) for x in val]
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            raise ValueError(f"[Error] Failed to parse list from string '{val}': {e}") from e
    return []


# ---------------------------------------------------------------------------
# Save & Load Dataset
# ---------------------------------------------------------------------------

def save_dataset(
    records: Union[List[Dict[str, Any]], pd.DataFrame],
    file_path: str,
    deduplicate_on: Optional[List[str]] = None,
    allow_unlabeled: bool = False,
) -> int:
    """
    Validates, deduplicates, and saves dataset records to a CSV file.

    Lists in 'token_probabilities' and 'entropy_values' are serialized to JSON strings
    to ensure lossless, safe round-trip persistence in CSV format.

    Args:
        records: A list of record dictionaries or a pandas DataFrame.
        file_path (str): Destination path for the CSV file.
        deduplicate_on (list[str], optional): Column names used to identify duplicates.
            Defaults to ['question', 'generated_answer'].
        allow_unlabeled (bool): If True, accepts label=-1 for unannotated data.

    Returns:
        int: Number of records saved.

    Raises:
        ValueError: If validation fails or conflicting duplicates are detected.
    """
    if deduplicate_on is None:
        deduplicate_on = ["question", "generated_answer"]

    # Convert list of dicts to DataFrame if needed
    if isinstance(records, list):
        df = pd.DataFrame(records)
    elif isinstance(records, pd.DataFrame):
        df = records.copy()
    else:
        raise TypeError("[Error] 'records' must be a list of dictionaries or a pandas DataFrame.")

    # Validate schema and values before processing
    validate_dataframe(df, allow_unlabeled=allow_unlabeled)

    initial_count = len(df)

    # Check for conflicting duplicates (same deduplicate_on keys but differing labels)
    existing_dup_cols = [c for c in deduplicate_on if c in df.columns]
    if existing_dup_cols and "label" in df.columns:
        label_variance = df.groupby(existing_dup_cols)["label"].nunique()
        conflicts = label_variance[label_variance > 1]
        if len(conflicts) > 0:
            raise ValueError(
                f"[Error] Found {len(conflicts)} conflicting duplicate record(s) with conflicting labels. "
                f"Conflicting duplicate records cannot be silently discarded."
            )

    # Deduplication for non-conflicting duplicates
    if existing_dup_cols:
        df = df.drop_duplicates(subset=existing_dup_cols, keep="first")
        removed_dups = initial_count - len(df)
        if removed_dups > 0:
            print(f"[Dataset] Removed {removed_dups} duplicate record(s) based on {existing_dup_cols}.")

    # Serialize list fields to JSON strings for safe CSV storage
    df_to_save = df.copy()
    df_to_save["token_probabilities"] = df_to_save["token_probabilities"].apply(_serialize_list_field)
    df_to_save["entropy_values"] = df_to_save["entropy_values"].apply(_serialize_list_field)

    # Ensure parent directory exists
    parent_dir = os.path.dirname(os.path.abspath(file_path))
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    df_to_save.to_csv(file_path, index=False, encoding="utf-8")
    print(f"[Dataset] Successfully saved {len(df_to_save)} record(s) to '{file_path}'.")

    return len(df_to_save)


def load_dataset(file_path: str, validate: bool = True, allow_unlabeled: bool = False) -> pd.DataFrame:
    """
    Loads a dataset from a CSV file, validates its structure, and deserializes
    token-level signal lists from JSON back into Python lists of floats.

    Args:
        file_path (str): Path to the CSV file to load.
        validate (bool): Whether to run schema/value validation after loading.
        allow_unlabeled (bool): Whether to permit unannotated records (label = -1).

    Returns:
        pd.DataFrame: Loaded DataFrame with restored Python lists for signals.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If validation fails or lists cannot be parsed.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[Error] Dataset file not found: '{file_path}'.")

    df = pd.read_csv(file_path, encoding="utf-8")

    if validate:
        validate_dataframe(df, allow_unlabeled=allow_unlabeled)

    # Deserialize JSON string columns back to Python lists of floats
    if "token_probabilities" in df.columns:
        df["token_probabilities"] = df["token_probabilities"].apply(_deserialize_list_field)

    if "entropy_values" in df.columns:
        df["entropy_values"] = df["entropy_values"].apply(_deserialize_list_field)

    print(f"[Dataset] Successfully loaded {len(df)} record(s) from '{file_path}'.")
    return df


# ---------------------------------------------------------------------------
# Self-Test / Verification Block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  ML Hallucination Detector — Dataset Manager Verification")
    print("=" * 65)

    test_csv_path = os.path.join("dataset", "sample_test_dataset.csv")

    # 1. Construct illustrative synthetic test records (NOT real research data)
    sample_records = [
        create_dataset_record(
            question="What is the capital of France?",
            generated_answer="The capital of France is Paris.",
            label=0,  # Factual
            average_probability=0.985,
            average_entropy=0.120,
            token_probabilities=[0.99, 0.98, 0.99, 0.98],
            entropy_values=[0.10, 0.12, 0.09, 0.17],
            source="illustrative_synthetic",
            reference_answer="Paris",
        ),
        create_dataset_record(
            question="Who walked on the Sun in 1982?",
            generated_answer="Neil Armstrong walked on the Sun in 1982.",
            label=1,  # Hallucinated
            average_probability=0.450,
            average_entropy=1.850,
            token_probabilities=[0.40, 0.42, 0.50, 0.48],
            entropy_values=[1.90, 1.80, 1.75, 1.95],
            source="illustrative_synthetic",
            reference_answer="Nobody has ever walked on the Sun.",
        ),
        # Duplicate record of the first entry to test deduplication
        create_dataset_record(
            question="What is the capital of France?",
            generated_answer="The capital of France is Paris.",
            label=0,
            average_probability=0.985,
            average_entropy=0.120,
            token_probabilities=[0.99, 0.98, 0.99, 0.98],
            entropy_values=[0.10, 0.12, 0.09, 0.17],
            source="illustrative_synthetic",
            reference_answer="Paris",
        ),
    ]

    print(f"\n[Test Setup] Created {len(sample_records)} illustrative records (including 1 intentional duplicate).")

    # Test 1: Deduplication and CSV Saving
    print("\n--- Test 1: Saving & Deduplication ---")
    saved_count = save_dataset(sample_records, test_csv_path)
    assert saved_count == 2, f"Expected 2 unique records saved, got {saved_count}"
    print("[PASS] Saving and deduplication verified (1 duplicate removed, 2 records saved).")

    # Test 2: CSV Loading & Round-trip Deserialization
    print("\n--- Test 2: Loading & Round-trip Deserialization ---")
    loaded_df = load_dataset(test_csv_path)
    assert len(loaded_df) == 2, f"Expected 2 records loaded, got {len(loaded_df)}"
    # Verify deserialization restored lists of floats
    first_probs = loaded_df.iloc[0]["token_probabilities"]
    first_ents = loaded_df.iloc[0]["entropy_values"]
    assert isinstance(first_probs, list) and isinstance(first_probs[0], float), "Token probabilities must be list of floats."
    assert isinstance(first_ents, list) and isinstance(first_ents[0], float), "Entropy values must be list of floats."
    print(f"       Record 1 Token Probs: {first_probs}")
    print(f"       Record 1 Entropy:     {first_ents}")
    print("[PASS] CSV loading and list deserialization verified.")

    # Test 3: Invalid Label Validation
    print("\n--- Test 3: Invalid Label Validation ---")
    try:
        invalid_record = [
            create_dataset_record(
                question="Q?",
                generated_answer="A.",
                label=2,  # Invalid! Must be 0 or 1
                average_probability=0.8,
                average_entropy=0.5,
                token_probabilities=[0.8],
                entropy_values=[0.5],
            )
        ]
        print("[FAIL] Invalid label was not caught!")
    except ValueError as e:
        print(f"[PASS] Invalid label successfully rejected with expected error:\n       {e}")

    # Test 4: Missing Column Detection
    print("\n--- Test 4: Missing Column Detection ---")
    incomplete_df = loaded_df.drop(columns=["average_entropy"])
    try:
        validate_dataframe(incomplete_df)
        print("[FAIL] Missing column was not caught!")
    except ValueError as e:
        print(f"[PASS] Missing column successfully detected with expected error:\n       {e}")

    # Test 5: Unannotated Record Handling (label = -1)
    print("\n--- Test 5: Unannotated Record Handling (label = -1) ---")
    unannotated_record = create_dataset_record(
        question="What is dark matter?",
        generated_answer="Dark matter is a hypothetical form of matter.",
        label=LABEL_UNLABELED,  # -1
        average_probability=0.75,
        average_entropy=0.85,
        token_probabilities=[0.75],
        entropy_values=[0.85],
        source="pilot_test",
        allow_unlabeled=True,
    )
    # Reject when allow_unlabeled=False (default behavior)
    try:
        save_dataset([unannotated_record], test_csv_path, allow_unlabeled=False)
        print("[FAIL] Unlabeled record was accepted under allow_unlabeled=False!")
    except ValueError:
        print("[PASS] Unlabeled record correctly rejected when allow_unlabeled=False.")

    # Accept when allow_unlabeled=True
    save_dataset([unannotated_record], test_csv_path, allow_unlabeled=True)
    unannotated_df = load_dataset(test_csv_path, allow_unlabeled=True)
    assert len(unannotated_df) == 1 and unannotated_df.iloc[0]["label"] == -1
    print("[PASS] Unlabeled record successfully saved and loaded when allow_unlabeled=True.")

    # Test 6: Safe Filter for Classifier Training (Excludes -1)
    print("\n--- Test 6: Classifier Training Filter (Excludes -1) ---")
    mixed_records = sample_records[:2] + [unannotated_record]
    mixed_df = pd.DataFrame(mixed_records)
    training_data = get_labeled_data(mixed_df)
    assert len(training_data) == 2, f"Expected 2 labeled records, got {len(training_data)}"
    assert -1 not in training_data["label"].values, "Unlabeled record leaked into training data!"
    print(f"[PASS] get_labeled_data() successfully isolated {len(training_data)} labeled records and excluded unannotated ones.")

    # Test 7: Conflicting Duplicate Detection
    print("\n--- Test 7: Conflicting Duplicate Detection ---")
    conflicting_records = [
        create_dataset_record(
            question="Is the earth round?",
            generated_answer="Yes, the earth is round.",
            label=0,
            average_probability=0.99,
            average_entropy=0.05,
            token_probabilities=[0.99],
            entropy_values=[0.05],
        ),
        create_dataset_record(
            question="Is the earth round?",
            generated_answer="Yes, the earth is round.",
            label=1,  # Conflict! Same question and answer, but conflicting label
            average_probability=0.99,
            average_entropy=0.05,
            token_probabilities=[0.99],
            entropy_values=[0.05],
        ),
    ]
    try:
        save_dataset(conflicting_records, test_csv_path)
        print("[FAIL] Conflicting duplicate was silently discarded!")
    except ValueError as e:
        print(f"[PASS] Conflicting duplicate successfully detected and rejected:\n       {e}")

    # Clean up test file
    if os.path.exists(test_csv_path):
        os.remove(test_csv_path)
        print(f"\n[Cleanup] Cleaned up temporary test file: '{test_csv_path}'")

    print("\n" + "=" * 65)
    print("  ALL DATASET MANAGER TESTS PASSED (Including Unlabeled Support)")
    print("=" * 65)
