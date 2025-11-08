#!/usr/bin/env python3
"""
Sample Evaluation Script for Fine-tuned LFM2-700M Mental Health Model

This script loads the fine-tuned model and runs inference on sample prompts
to evaluate the quality of responses.

WARNING: This is for research purposes only. Outputs should not be used
as a substitute for professional mental health support.
"""

import argparse
import json
import os
import sys
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned model on sample prompts"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (default: auto-detect latest)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="/workspace/data/samples/seed_eval.jsonl",
        help="Path to sample prompts JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL file for results (default: auto-generate)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter (default: 0.9)",
    )
    return parser.parse_args()


def find_latest_checkpoint():
    """Find the most recent 'final' checkpoint."""
    sft_dir = Path("/workspace/outputs/sft")
    if not sft_dir.exists():
        return None

    # Look for 'final' directories
    final_dirs = list(sft_dir.glob("*/final"))
    if final_dirs:
        # Sort by modification time
        final_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(final_dirs[0])

    # Fallback: look for any checkpoint
    checkpoint_dirs = list(sft_dir.glob("*/checkpoint-*"))
    if checkpoint_dirs:
        checkpoint_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(checkpoint_dirs[0])

    return None


def load_model(checkpoint_path: str):
    """Load the fine-tuned model and tokenizer."""
    print("=" * 80)
    print("Loading Model")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "LiquidAI/LFM2-700M",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Loading adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_path,
        torch_dtype=torch.bfloat16,
    )

    # Merge adapter for faster inference
    print("Merging adapter with base model...")
    model = model.merge_and_unload()

    model.eval()

    print(f"✅ Model loaded successfully")
    print(f"   Device: {next(model.parameters()).device}")
    print()

    return model, tokenizer


def load_samples(samples_path: str):
    """Load evaluation samples from JSONL file."""
    samples = []
    with open(samples_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def format_prompt(sample: dict, tokenizer) -> str:
    """Format a sample into a prompt for the model."""
    if "messages" in sample:
        # Use chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                sample["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback formatting
            messages = sample["messages"]
            parts = []
            for msg in messages:
                role = msg["role"].capitalize()
                content = msg["content"]
                parts.append(f"{role}: {content}")
            prompt = "\n".join(parts) + "\nAssistant:"
    elif "prompt" in sample:
        prompt = sample["prompt"]
    else:
        raise ValueError("Sample must have 'messages' or 'prompt' field")

    return prompt


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a response for the given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part (skip input)
    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response.strip()


def main():
    args = parse_args()

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        print("No checkpoint specified, searching for latest...")
        checkpoint_path = find_latest_checkpoint()

        if not checkpoint_path:
            print("❌ No checkpoint found")
            print("\nPlease specify a checkpoint with --checkpoint, or train a model first.")
            sys.exit(1)

        print(f"Found checkpoint: {checkpoint_path}\n")

    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint does not exist: {checkpoint_path}")
        sys.exit(1)

    # Load model
    model, tokenizer = load_model(checkpoint_path)

    # Load samples
    print("=" * 80)
    print("Loading Evaluation Samples")
    print("=" * 80)

    if not Path(args.samples).exists():
        print(f"❌ Samples file does not exist: {args.samples}")
        sys.exit(1)

    samples = load_samples(args.samples)
    print(f"Loaded {len(samples)} samples from {args.samples}")
    print()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        job_name = Path(checkpoint_path).parent.name
        output_dir = Path("/workspace/outputs/eval")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job_name}.jsonl"

    # Run evaluation
    print("=" * 80)
    print("Running Evaluation")
    print("=" * 80)
    print(f"Output: {output_path}")
    print()

    results = []

    for idx, sample in enumerate(samples, 1):
        print(f"[{idx}/{len(samples)}] Generating response...")

        try:
            prompt = format_prompt(sample, tokenizer)
            response = generate_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            result = {
                "sample_id": idx,
                "prompt": sample.get("prompt") or sample.get("messages", [{}])[-1].get("content", ""),
                "response": response,
                "metadata": sample.get("metadata", {}),
                "generation_params": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                },
            }

            results.append(result)

            # Print preview
            print(f"  Prompt: {result['prompt'][:100]}...")
            print(f"  Response: {response[:150]}...")
            print()

        except Exception as e:
            print(f"  ❌ Error generating response: {e}")
            results.append({
                "sample_id": idx,
                "error": str(e),
            })

    # Save results
    print("=" * 80)
    print("Saving Results")
    print("=" * 80)

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"✅ Results saved to: {output_path}")
    print(f"   Total samples: {len(results)}")
    print()

    # Print summary
    print("=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print()
    print("Sample Responses:")
    print("-" * 80)

    for result in results[:3]:  # Show first 3
        if "error" not in result:
            print(f"\nPrompt: {result['prompt']}")
            print(f"Response: {result['response']}")
            print("-" * 80)

    print()
    print("⚠️  IMPORTANT DISCLAIMER:")
    print("This model is for RESEARCH PURPOSES ONLY.")
    print("Outputs should NOT be used as a substitute for professional mental health support.")
    print("If you or someone you know is in crisis, please contact:")
    print("  - National Suicide Prevention Lifeline: 988 (US)")
    print("  - Crisis Text Line: Text HOME to 741741")
    print("  - International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/")
    print()


if __name__ == "__main__":
    main()
