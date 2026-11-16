# ML Hallucination Detector

A research-oriented Machine Learning project designed to detect potential hallucinations in Large Language Models (LLMs) by analyzing their internal signals, focusing on:

1. **Token Probability Distributions**: Monitoring next-token likelihood and output distribution characteristics.
2. **Predictive Entropy**: Measuring uncertainty and dispersion across generated token candidates.
3. **Hidden-State Representations**: Investigating internal layer activation geometries and latent representations to identify divergence from factual groundings.

Initial experiments utilize the `Qwen/Qwen2.5-0.5B-Instruct` model via Hugging Face Transformers and PyTorch.
