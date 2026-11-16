"""
model_loader.py
---------------
Handles loading the Qwen/Qwen2.5-0.5B-Instruct model and its tokenizer,
and provides a basic text generation function.

This is Phase 1 of the ML Hallucination Detector project.
No hallucination detection, entropy extraction, or hidden-state analysis
is implemented here — this module is intentionally kept minimal.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Maximum number of NEW tokens the model is allowed to generate per response.
MAX_NEW_TOKENS = 150


# ---------------------------------------------------------------------------
# Device Selection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """
    Automatically selects the best available compute device.

    Returns:
        torch.device: 'cuda' if an NVIDIA GPU is available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] GPU detected: {torch.cuda.get_device_name(0)}. Using CUDA.")
    else:
        device = torch.device("cpu")
        print("[Device] No GPU detected. Using CPU.")
    return device


# ---------------------------------------------------------------------------
# Model & Tokenizer Loading
# ---------------------------------------------------------------------------

def load_model() -> tuple:
    """
    Downloads (if not cached) and loads the Qwen2.5-0.5B-Instruct model
    and its tokenizer from Hugging Face.

    Steps performed:
        1. Detects the best available device (CUDA or CPU).
        2. Loads the tokenizer — handles converting text <-> token IDs.
        3. Loads the causal language model — the neural network itself.
        4. Moves the model to the selected device.
        5. Puts the model in evaluation mode (disables dropout, etc.).

    Returns:
        tuple: (model, tokenizer)
            - model (AutoModelForCausalLM): The loaded LLM ready for inference.
            - tokenizer (AutoTokenizer): The corresponding tokenizer.

    Raises:
        RuntimeError: If the model or tokenizer cannot be loaded.
    """
    device = get_device()

    print(f"[Model] Loading tokenizer for '{MODEL_ID}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,   # Qwen models may require this
        )
    except Exception as e:
        raise RuntimeError(f"[Error] Failed to load tokenizer: {e}") from e

    print(f"[Model] Tokenizer loaded successfully.")

    print(f"[Model] Loading model '{MODEL_ID}'... (this may take a moment on first run)")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            # float16 halves memory usage on GPU; CPU needs float32 for stability
            # Note: transformers >=5.x uses 'dtype' instead of 'torch_dtype'
            trust_remote_code=True,
        )
    except Exception as e:
        raise RuntimeError(f"[Error] Failed to load model: {e}") from e

    # Move all model weights to the selected device (GPU or CPU)
    model = model.to(device)

    # Evaluation mode: disables training-specific behaviour like dropout.
    # This is important for reproducible, deterministic inference.
    model.eval()

    print(f"[Model] Model loaded and set to evaluation mode on '{device}'.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Response Generation
# ---------------------------------------------------------------------------

def generate_response(model, tokenizer, question: str) -> str:
    """
    Generates a text response from the model for a given user question.

    How it works:
        1. Formats the raw question into the Qwen chat template
           (adds system/user/assistant role markers the model expects).
        2. Tokenizes the formatted prompt into integer token IDs.
        3. Runs a forward pass + greedy decoding (do_sample=False) to get
           the most likely continuation token-by-token.
        4. Decodes only the newly generated tokens back into plain text.

    Args:
        model  (AutoModelForCausalLM): A loaded model returned by load_model().
        tokenizer (AutoTokenizer):     The corresponding tokenizer.
        question (str):                The raw user question / prompt.

    Returns:
        str: The model's generated answer as a plain text string.

    Raises:
        ValueError: If the question is empty or not a string.
        RuntimeError: If generation fails unexpectedly.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("[Error] 'question' must be a non-empty string.")

    # ------------------------------------------------------------------
    # Step 1 — Apply the model's built-in chat template
    # ------------------------------------------------------------------
    # Qwen models are instruction-tuned and expect a specific prompt format
    # with [INST] / <|im_start|> markers. apply_chat_template handles this
    # automatically so we never need to hard-code the format ourselves.
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful, accurate, and concise assistant. "
                "Answer questions factually and admit uncertainty when unsure."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    print(f"[Generation] Formatting prompt using chat template...")
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,          # Return a plain string first; we tokenize next
        add_generation_prompt=True,  # Appends the assistant turn opener
    )

    # ------------------------------------------------------------------
    # Step 2 — Tokenize the formatted prompt
    # ------------------------------------------------------------------
    # Determine which device the model lives on so inputs go to the same place
    device = next(model.parameters()).device

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",   # Return PyTorch tensors
    ).to(device)

    input_length = inputs["input_ids"].shape[1]  # Number of prompt tokens

    # ------------------------------------------------------------------
    # Step 3 — Generate tokens (greedy / deterministic)
    # ------------------------------------------------------------------
    print(f"[Generation] Running inference (max {MAX_NEW_TOKENS} new tokens)...")
    try:
        with torch.no_grad():
            # torch.no_grad() tells PyTorch not to build a gradient graph,
            # which saves memory and speeds up inference significantly.
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,        # Greedy decoding — always picks the
                                        # highest-probability next token.
                                        # Deterministic and reproducible.
                temperature=1.0,        # Neutral temperature (not used when
                                        # do_sample=False, but set for clarity)
                pad_token_id=tokenizer.eos_token_id,  # Suppress padding warnings
            )
    except Exception as e:
        raise RuntimeError(f"[Error] Text generation failed: {e}") from e

    # ------------------------------------------------------------------
    # Step 4 — Decode only the NEW tokens (skip the input prompt)
    # ------------------------------------------------------------------
    # output_ids contains prompt tokens + generated tokens.
    # We slice off the prompt portion so the returned string is just the answer.
    generated_ids = output_ids[0][input_length:]
    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,   # Remove <|im_end|> and similar markers
    ).strip()

    print(f"[Generation] Response generated successfully.")
    return answer


# ---------------------------------------------------------------------------
# Quick Smoke Test (run this file directly to verify everything works)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  ML Hallucination Detector — Model Loader Smoke Test")
    print("=" * 60)

    model, tokenizer = load_model()

    test_question = "What is the capital of France?"
    print(f"\n[Test] Question: {test_question}\n")

    answer = generate_response(model, tokenizer, test_question)

    print(f"\n[Test] Answer:\n{answer}")
    print("\n[Smoke test complete]")
