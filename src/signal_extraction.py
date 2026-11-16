"""
signal_extraction.py
--------------------
Extracts internal uncertainty signals from the Qwen2.5-0.5B-Instruct model
during text generation.

Signals extracted per generated token:
  1. Token probability  — how confident the model was in the token it picked.
  2. Predictive entropy — how "spread out" the full probability distribution
                         was at that step (high = uncertain, low = confident).

This is Phase 2 of the ML Hallucination Detector project.
No classifier or hallucination decision is made here.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: token-level probability extraction
# ---------------------------------------------------------------------------

def extract_token_probabilities(generation_scores: tuple, generated_tokens: torch.Tensor) -> list[float]:
    """
    Extracts the probability of each actually selected token from the corresponding
    generation score distribution at each step.

    Background:
        - `generation_scores` is a tuple with one logit tensor per generated token.
        - Each tensor has shape (batch_size, vocab_size).
        - `generated_tokens` is a 1D tensor containing the exact token IDs chosen
          by the model at each step (excluding prompt tokens).
        - For each step, we apply softmax over the logits to obtain the probability
          distribution over the entire vocabulary, and extract the probability
          corresponding to the actual selected token ID.

    Args:
        generation_scores (tuple[Tensor]): Per-step logit tensors from model.generate().
                                           Length == number of generated tokens.
        generated_tokens (Tensor): 1D tensor of token IDs actually selected at each step.

    Returns:
        list[float]: Probability of the actual generated token at each step (0.0 – 1.0).
    """
    token_probabilities = []

    for step_idx, step_logits in enumerate(generation_scores):
        token_id = generated_tokens[step_idx].item()

        # step_logits shape: (batch_size=1, vocab_size)
        # Softmax converts raw logits → probability distribution (sums to 1.0).
        probs = F.softmax(step_logits, dim=-1)

        # Extract probability of the specific token actually generated at this step
        chosen_prob = probs[0, token_id].item()
        token_probabilities.append(chosen_prob)

    return token_probabilities


# ---------------------------------------------------------------------------
# Helper: predictive entropy
# ---------------------------------------------------------------------------

def compute_entropy_values(generation_scores: tuple) -> list[float]:
    """
    Computes Shannon entropy of the token probability distribution across the
    complete vocabulary at each generation step.

    Formula:
        H = -Σ  p(x) · log( p(x) )      (summed over all vocabulary tokens)

    Interpretation:
        - Low entropy  → the model is very confident (probability concentrated).
        - High entropy → the model is uncertain (probability spread across many candidates).

    Numerical stability:
        Uses log_softmax and handles the p=0 boundary condition cleanly:
        lim_{p->0} p · log(p) = 0.
        This avoids both log(0) -> -inf and artificial probability distortion from clamping.

    Args:
        generation_scores (tuple[Tensor]): Scores from model.generate().

    Returns:
        list[float]: One entropy value (in nats) per generated token.
    """
    entropy_values = []

    for step_logits in generation_scores:
        # Convert logits → probabilities and log-probabilities stably
        probs = F.softmax(step_logits, dim=-1)          # shape: (1, vocab_size)
        log_probs = F.log_softmax(step_logits, dim=-1)  # shape: (1, vocab_size)

        # H = -Σ p · log(p)
        # For tokens where prob == 0, p * log(p) is 0 by mathematical limit
        p_log_p = torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs))
        entropy = -p_log_p.sum(dim=-1)                  # shape: (1,)

        entropy_values.append(entropy.item())

    return entropy_values


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def extract_signals(model, tokenizer, question: str) -> dict:
    """
    Runs a full forward pass for a given question and extracts token-level
    uncertainty signals from the model's generation process.

    Pipeline:
        1. Format question using the model's chat template.
        2. Tokenize and move tensors to the model's device.
        3. Generate a response with output_scores=True so we get per-step
           logits alongside the generated token IDs.
        4. Extract token probabilities and entropy from those scores.
        5. Decode the generated answer (new tokens only).
        6. Return everything in a structured dictionary.

    Args:
        model      : Loaded AutoModelForCausalLM (from model_loader.py).
        tokenizer  : Matching AutoTokenizer.
        question (str): Raw user question.

    Returns:
        dict with keys:
            "answer"               – generated text (str)
            "token_probabilities"  – list[float], one per generated token
            "entropy_values"       – list[float], one per generated token
            "average_probability"  – float, mean of token_probabilities
            "average_entropy"      – float, mean of entropy_values

    Raises:
        ValueError:   If question is empty or not a string.
        RuntimeError: If generation fails.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("[Error] 'question' must be a non-empty string.")

    # ------------------------------------------------------------------
    # Step 1 — Apply chat template
    # ------------------------------------------------------------------
    # Qwen2.5-Instruct was trained with a specific conversation format.
    # apply_chat_template wraps the question in the right role markers.
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful, accurate, and concise assistant. "
                "Answer questions factually and admit uncertainty when unsure."
            ),
        },
        {"role": "user", "content": question},
    ]

    print("[Signal] Formatting prompt via chat template...")
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,            # Return a plain string first
        add_generation_prompt=True,
    )

    # ------------------------------------------------------------------
    # Step 2 — Tokenize & send to the model's device
    # ------------------------------------------------------------------
    device = next(model.parameters()).device  # wherever the model lives

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",   # PyTorch tensors
    ).to(device)

    input_length = inputs["input_ids"].shape[1]  # number of prompt tokens

    # ------------------------------------------------------------------
    # Step 3 — Generate with scores enabled
    # ------------------------------------------------------------------
    # Key flags:
    #   return_dict_in_generate=True  → result is a named object, not a plain tensor
    #   output_scores=True            → generation_scores is included in the result
    #   do_sample=False               → greedy decoding (deterministic)
    #
    # generation_scores is a tuple of tensors, one per generated token.
    # Each tensor: shape (batch_size, vocab_size) = (1, ~151936 for Qwen).

    print("[Signal] Running generation with output_scores=True ...")
    try:
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,   # <-- needed to access scores
                output_scores=True,             # <-- gives us per-step logits
            )
    except Exception as exc:
        raise RuntimeError(f"[Error] Generation failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Step 4 — Extract generated tokens (prompt tokens sliced off)
    # ------------------------------------------------------------------
    # generation_output.sequences contains prompt tokens + newly generated tokens.
    # We strictly slice off the prompt tokens so only generated tokens remain.
    generated_ids = generation_output.sequences[0][input_length:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    scores = generation_output.scores
    if not scores:
        raise RuntimeError("[Error] No generation scores were returned. "
                           "Ensure output_scores=True was set.")

    # Verification: Each generated token must have a corresponding generation score tensor
    if len(scores) != len(generated_ids):
        raise RuntimeError(
            f"[Error] Score/token count mismatch: {len(scores)} score tensors "
            f"vs {len(generated_ids)} generated tokens."
        )

    # ------------------------------------------------------------------
    # Step 5 — Extract per-token probability and predictive entropy
    # ------------------------------------------------------------------
    print(f"[Signal] Extracting probabilities for {len(generated_ids)} generated token(s)...")
    token_probabilities = extract_token_probabilities(scores, generated_ids)

    print(f"[Signal] Computing entropy for {len(scores)} generated token(s)...")
    entropy_values = compute_entropy_values(scores)

    # ------------------------------------------------------------------
    # Step 6 — Aggregate statistics
    # ------------------------------------------------------------------
    num_tokens = len(token_probabilities)

    average_probability = (
        sum(token_probabilities) / num_tokens if num_tokens > 0 else 0.0
    )
    average_entropy = (
        sum(entropy_values) / num_tokens if num_tokens > 0 else 0.0
    )

    print("[Signal] Signal extraction complete.")

    return {
        "answer":               answer,
        "token_probabilities":  token_probabilities,
        "entropy_values":       entropy_values,
        "average_probability":  average_probability,
        "average_entropy":      average_entropy,
    }


# ---------------------------------------------------------------------------
# Quick Test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    # Add the project root to the Python path so we can import model_loader
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.model_loader import load_model

    print("=" * 65)
    print("  ML Hallucination Detector — Signal Extraction Test")
    print("=" * 65)

    # Load the model (uses cached weights — fast after first run)
    model, tokenizer = load_model()

    test_question = "What is the capital of India?"
    print(f"\n[Test] Question: {test_question}\n")

    # Extract signals
    signals = extract_signals(model, tokenizer, test_question)

    # ----------------------------------------------------------------
    # Print results
    # ----------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  RESULTS")
    print("=" * 65)

    print(f"\n1. Generated Answer:\n   {signals['answer']}")

    print(f"\n2. Token Probabilities ({len(signals['token_probabilities'])} tokens):")
    for i, prob in enumerate(signals["token_probabilities"]):
        print(f"   Token {i+1:>3}: {prob:.6f}")

    print(f"\n3. Entropy Values ({len(signals['entropy_values'])} tokens):")
    for i, ent in enumerate(signals["entropy_values"]):
        print(f"   Token {i+1:>3}: {ent:.6f} nats")

    print(f"\n4. Average Token Probability : {signals['average_probability']:.6f}")
    print(f"5. Average Entropy           : {signals['average_entropy']:.6f} nats")

    print("\n[Test complete]")
