#!/usr/bin/env python3
"""
Test VLLM output space sampling with language-based interaction like chat.py
"""

import sys
sys.path.insert(0, '.')

from gpt_oss.vllm.token_generator import TokenGenerator

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    StreamState,
    SystemContent,
    TextContent,
    ToolDescription,
    load_harmony_encoding,
)

def test_language_sampling():
    print("Testing VLLM Output Space Sampling with Language")
    print("=" * 55)

    try:
        # Load encoding like in chat.py
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        print("✓ Harmony encoding loaded")

        # Create generator with sampling enabled
        generator = TokenGenerator(
            "gpt-oss-20b",
            tensor_parallel_size=1,
            enable_output_sampling=True,
            noise_std=0.15,
            noise_spread=0.3
        )
        print("✓ Generator created with sampling enabled")

        # Test prompt about dog care website colors
        prompt = "Suggest 3 colors for a dog care website:"
        prompt_tokens = encoding.encode(prompt)
        stop_tokens = encoding.stop_tokens()

        print(f"\nPrompt: '{prompt}'")
        print(f"Encoded to {len(prompt_tokens)} tokens")

        # Generate with sampling enabled
        print("\n--- Generation WITH sampling ---")
        sampled_tokens = []
        for i, token in enumerate(generator.generate(
            prompt_tokens=prompt_tokens,
            stop_tokens=stop_tokens,
            max_tokens=50,
            temperature=0.8
        )):
            sampled_tokens.append(token)
            if i >= 49:
                break

        sampled_text = encoding.decode(sampled_tokens)
        print(f"Output: {sampled_text}")

        # Generate with sampling disabled
        print("\n--- Generation WITHOUT sampling ---")
        generator.set_output_sampling(False)
        normal_tokens = []
        for i, token in enumerate(generator.generate(
            prompt_tokens=prompt_tokens,
            stop_tokens=stop_tokens,
            max_tokens=50,
            temperature=0.8
        )):
            normal_tokens.append(token)
            if i >= 49:
                break

        normal_text = encoding.decode(normal_tokens)
        print(f"Output: {normal_text}")

        # Compare results
        print("\n--- Comparison ---")
        print(f"Outputs are {'different' if sampled_text != normal_text else 'identical'}")

        if sampled_text != normal_text:
            print("✓ Output space sampling is working!")
            print(f"Sampled length: {len(sampled_tokens)} tokens")
            print(f"Normal length: {len(normal_tokens)} tokens")
        else:
            print("⚠ Outputs identical - sampling may need adjustment")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_prompts():
    """Test sampling with different prompts to show variety."""
    print("\n" + "=" * 55)
    print("Testing Multiple Prompts with Sampling")
    print("=" * 55)

    try:
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        generator = TokenGenerator(
            "gpt-oss-20b",
            tensor_parallel_size=1,
            enable_output_sampling=True,
            noise_std=0.2,  # Higher noise for more variation
            noise_spread=0.4
        )

        prompts = [
            "What are the best dog breeds for families?",
            "How to train a puppy:",
            "Dog nutrition tips:"
        ]

        for i, prompt in enumerate(prompts, 1):
            print(f"\n--- Test {i}: {prompt} ---")
            prompt_tokens = encoding.encode(prompt)
            stop_tokens = encoding.get_stop_tokens()

            tokens = []
            for j, token in enumerate(generator.generate(
                prompt_tokens=prompt_tokens,
                stop_tokens=stop_tokens,
                max_tokens=30,
                temperature=0.9
            )):
                tokens.append(token)
                if j >= 29:
                    break

            response = encoding.decode(tokens)
            print(f"Response: {response}")

        print("\n✓ Multiple prompt test completed")

    except Exception as e:
        print(f"✗ Multiple prompt test failed: {e}")

if __name__ == "__main__":
    # Run main language test
    success = test_language_sampling()

    # Run multiple prompts test if main test succeeded
    if success:
        test_multiple_prompts()

    print("\n" + "=" * 55)
    print("Testing complete!")
