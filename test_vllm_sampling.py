#!/usr/bin/env python3
"""
Simple test for VLLM output space sampling.
"""

import sys
sys.path.insert(0, '.')

from gpt_oss.vllm.token_generator import TokenGenerator

def test_vllm_sampling():
    print("Testing VLLM output space sampling...")

    # Create generator with sampling enabled
    try:
        generator = TokenGenerator(
            "gpt-oss-20b",  # Remove trailing slash
            tensor_parallel_size=1,
            enable_output_sampling=True,
            noise_std=0.15,
            noise_spread=0.3
        )
        print("✓ Generator created successfully with sampling enabled")

        # Simple tokenization (basic approach)
        prompt = "Hello world"
        prompt_tokens = [10, 20, 30]  # Dummy tokens for testing

        print(f"Testing generation with prompt tokens: {prompt_tokens}")

        # Generate a few tokens
        tokens = []
        for i, token in enumerate(generator.generate(
            prompt_tokens=prompt_tokens,
            stop_tokens=[50256],  # Common EOS token
            max_tokens=5,
            temperature=0.7
        )):
            tokens.append(token)
            print(f"Generated token {i+1}: {token}")
            if i >= 4:  # Limit to 5 tokens
                break

        print(f"✓ Generated tokens with sampling: {tokens}")

        # Test disabling sampling
        generator.set_output_sampling(False)
        print("✓ Sampling disabled")

        # Generate again without sampling
        normal_tokens = []
        for i, token in enumerate(generator.generate(
            prompt_tokens=prompt_tokens,
            stop_tokens=[50256],
            max_tokens=5,
            temperature=0.7
        )):
            normal_tokens.append(token)
            if i >= 4:
                break

        print(f"✓ Generated tokens without sampling: {normal_tokens}")

        # Compare results
        if tokens != normal_tokens:
            print("✓ Sampling appears to be working - outputs differ!")
        else:
            print("⚠ Warning: Outputs are identical - sampling may not be active")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vllm_sampling()