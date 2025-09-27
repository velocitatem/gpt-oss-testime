#!/usr/bin/env python3
"""
Test script for output space sampling functionality in GPT-OSS model.
"""

import torch
from gpt_oss.torch.model import TokenGenerator
from gpt_oss.harmony.encoding import load_harmony_encoding, HarmonyEncodingName

def test_output_sampling():
    """Test the output space sampling functionality."""

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load encoding
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Create generator with output sampling enabled
    print("Creating TokenGenerator with output sampling...")
    generator = TokenGenerator(
        checkpoint="gpt-oss-20b/",
        device=device,
        enable_output_sampling=True,
        noise_std=0.15,      # Adjust noise strength
        noise_spread=0.3     # Adjust spread around center
    )

    # Test prompt
    prompt = "What is the meaning of life?"
    prompt_tokens = encoding.encode(prompt)
    stop_tokens = encoding.get_stop_tokens()

    print(f"Prompt: {prompt}")
    print(f"Prompt tokens: {prompt_tokens}")

    # Generate with sampling enabled
    print("\n=== Generation with sampling ENABLED ===")
    generator.set_output_sampling(True)
    sampled_tokens = []
    for i, token in enumerate(generator.generate(
        prompt_tokens=prompt_tokens,
        stop_tokens=stop_tokens,
        max_tokens=50,
        temperature=0.7,
        enable_sampling=True
    )):
        sampled_tokens.append(token)
        if i >= 49:  # Limit output
            break

    sampled_text = encoding.decode(sampled_tokens)
    print(f"Sampled output: {sampled_text}")

    # Generate with sampling disabled for comparison
    print("\n=== Generation with sampling DISABLED ===")
    generator.set_output_sampling(False)
    normal_tokens = []
    for i, token in enumerate(generator.generate(
        prompt_tokens=prompt_tokens,
        stop_tokens=stop_tokens,
        max_tokens=50,
        temperature=0.7,
        enable_sampling=False
    )):
        normal_tokens.append(token)
        if i >= 49:  # Limit output
            break

    normal_text = encoding.decode(normal_tokens)
    print(f"Normal output: {normal_text}")

    # Compare outputs
    print("\n=== COMPARISON ===")
    print(f"Tokens differ: {sampled_tokens != normal_tokens}")
    print(f"Text differs: {sampled_text != normal_text}")

def test_different_noise_levels():
    """Test different noise levels and spreads."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    prompt = "The future of AI is"
    prompt_tokens = encoding.encode(prompt)
    stop_tokens = encoding.get_stop_tokens()

    noise_configs = [
        (0.05, 0.1),  # Low noise, tight spread
        (0.15, 0.3),  # Medium noise, medium spread
        (0.3, 0.5),   # High noise, wide spread
    ]

    for noise_std, noise_spread in noise_configs:
        print(f"\n=== Testing noise_std={noise_std}, noise_spread={noise_spread} ===")

        generator = TokenGenerator(
            checkpoint="gpt-oss-20b/",
            device=device,
            enable_output_sampling=True,
            noise_std=noise_std,
            noise_spread=noise_spread
        )

        tokens = []
        for i, token in enumerate(generator.generate(
            prompt_tokens=prompt_tokens,
            stop_tokens=stop_tokens,
            max_tokens=30,
            temperature=0.7,
            enable_sampling=True
        )):
            tokens.append(token)
            if i >= 29:
                break

        text = encoding.decode(tokens)
        print(f"Output: {text}")

if __name__ == "__main__":
    print("Testing GPT-OSS Output Space Sampling")
    print("=" * 50)

    try:
        test_output_sampling()
        print("\n" + "=" * 50)
        test_different_noise_levels()
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()