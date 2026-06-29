"""
Base vs fine-tuned comparison for the JSX/React style fine-tune.

Runs both the original Qwen2.5-Coder-1.5B-Instruct and your LoRA adapter on
the same held-out component-generation prompts, then scores each completion
against the concrete style patterns extracted from your real code (Tailwind
arbitrary values, prop-destructuring, sparse comments, etc.) so you get an
objective signal on top of the side-by-side text.

Run with:
    modal run eval_jsx_compare.py

Requires train_jsx.py to have already run and saved the adapter to
/data/outputs/jsx-final-adapters on the jsx-finetune-vol volume.

Output: a markdown report on the volume + printed directly to logs.
Pull it down with:
    modal volume get jsx-finetune-vol /outputs/jsx_comparison_report.md ./jsx_comparison_report.md
"""

import modal
import re

app = modal.App("jsx-style-finetune-eval")

volume = modal.Volume.from_name("jsx-finetune-vol", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.0",
    "transformers==4.46.3",
    "peft==0.13.2",
    "bitsandbytes==0.44.1",
    "accelerate==1.0.1",
    "huggingface_hub==0.26.2",
)

VOLUME_PATH = "/data"
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = f"{VOLUME_PATH}/outputs/jsx-final-adapters"

# Held-out prompts the model wasn't trained on directly — different
# component types than the 20 training tasks, to test generalization
# of the STYLE rather than memorization of specific components.
EVAL_PROMPTS = [
    "Write a React component for a 'like' button that shows a heart icon and a count, toggling filled/outline on click.",
    "Write a React/Next.js component for a comment box with a textarea and a submit button.",
    "Write a React component for a progress bar that fills based on a percentage prop.",
    "Write a React component for a navbar with a logo on the left and nav links on the right.",
    "Write a React component for an empty-state message shown when a list has no items, with an icon and a short message.",
    "Write a React component for a badge/label that shows one of three priority levels: low, medium, high.",
]

# The concrete style patterns extracted from Sope's real code — used here
# to score completions, not just eyeball them.
STYLE_CHECKS = {
    "tailwind_arbitrary_values": re.compile(r'\[\d+(\.\d+)?(px|%|rem)\]'),
    "typed_props_destructured": re.compile(r'const\s+\w+\s*=\s*\(\s*\{[^}]*\}\s*:\s*\w*Props'),
    "default_export_at_bottom": re.compile(r'export default \w+;?\s*$'),
    "react_fc_typing": re.compile(r'React\.FC<'),
    "union_type_for_constrained_prop": re.compile(r'type\s+\w+\s*=\s*\n?\s*\|?\s*"[^"]+"\s*\|'),
}


def score_style(code: str) -> dict:
    """Returns which style patterns are present in a given completion."""
    return {name: bool(pattern.search(code)) for name, pattern in STYLE_CHECKS.items()}


@app.function(
    image=image,
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=1800,
)
def run_comparison():
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    from huggingface_hub import login

    login(token=os.environ["HF_TOKEN"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def generate(model, prompt, max_new_tokens=400):
        messages = [
            {"role": "system", "content": "You are an assistant that writes React/Next.js components matching the user's personal coding style: Tailwind arbitrary values, sparse comments, typed Props destructured in the function signature, default export at the bottom of the file."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    # ── Base model first ─────────────────────────────────────────
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )

    print(f"Generating {len(EVAL_PROMPTS)} base-model completions...")
    base_outputs = []
    for i, prompt in enumerate(EVAL_PROMPTS):
        print(f"  base [{i+1}/{len(EVAL_PROMPTS)}]")
        base_outputs.append(generate(base_model, prompt))

    del base_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("Base model freed from memory.")

    # ── Base + adapter ────────────────────────────────────────────
    print("Loading base model again for adapter...")
    adapter_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    print(f"Attaching LoRA adapter from {ADAPTER_PATH}...")
    finetuned_model = PeftModel.from_pretrained(adapter_base, ADAPTER_PATH)

    print(f"Generating {len(EVAL_PROMPTS)} fine-tuned completions...")
    finetuned_outputs = []
    for i, prompt in enumerate(EVAL_PROMPTS):
        print(f"  finetuned [{i+1}/{len(EVAL_PROMPTS)}]")
        finetuned_outputs.append(generate(finetuned_model, prompt))

    # ── Score every completion against the real style patterns ─────
    base_scores = [score_style(c) for c in base_outputs]
    finetuned_scores = [score_style(c) for c in finetuned_outputs]

    pattern_names = list(STYLE_CHECKS.keys())
    base_totals = {p: sum(s[p] for s in base_scores) for p in pattern_names}
    finetuned_totals = {p: sum(s[p] for s in finetuned_scores) for p in pattern_names}

    # ── Build markdown report ───────────────────────────────────────
    lines = [
        "# JSX Style Fine-Tune: Base vs Fine-Tuned Comparison",
        "",
        f"Base model: `{MODEL_ID}`",
        "Adapter: LoRA, r=16, alpha=32, ~21 training examples (5 real + synthetic), 8 epochs",
        "",
        "Note: temperature=0.2 sampling, so re-running produces slightly different",
        "text each time. Judge the overall pattern across prompts, not one completion.",
        "",
        "## Style Pattern Score Summary",
        "",
        f"Out of {len(EVAL_PROMPTS)} held-out prompts, how many completions matched each style pattern:",
        "",
        "| Pattern | Base model | Fine-tuned |",
        "|---|---|---|",
    ]
    for p in pattern_names:
        lines.append(f"| {p} | {base_totals[p]}/{len(EVAL_PROMPTS)} | {finetuned_totals[p]}/{len(EVAL_PROMPTS)} |")

    lines += [
        "",
        "A higher fine-tuned number than base, on patterns we explicitly trained for,",
        "is evidence the fine-tune actually shifted style — not just a vibe.",
        "",
        "---",
        "",
    ]

    for i, prompt in enumerate(EVAL_PROMPTS):
        lines.append(f"## Prompt {i+1}")
        lines.append("")
        lines.append(f"**Prompt:** {prompt}")
        lines.append("")
        lines.append(f"**Base model style matches:** {[p for p in pattern_names if base_scores[i][p]]}")
        lines.append("**Base model:**")
        lines.append("```")
        lines.append(base_outputs[i])
        lines.append("```")
        lines.append("")
        lines.append(f"**Fine-tuned style matches:** {[p for p in pattern_names if finetuned_scores[i][p]]}")
        lines.append("**Fine-tuned model:**")
        lines.append("```")
        lines.append(finetuned_outputs[i])
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    report = "\n".join(lines)

    report_path = f"{VOLUME_PATH}/outputs/jsx_comparison_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    volume.commit()

    print(f"\nReport written to volume at {report_path}")
    print("Pull it down with:")
    print(f"  modal volume get jsx-finetune-vol /outputs/jsx_comparison_report.md ./jsx_comparison_report.md")

    print("\n" + "=" * 80)
    print("STYLE PATTERN SCORE SUMMARY")
    print("=" * 80)
    for p in pattern_names:
        print(f"  {p}: base={base_totals[p]}/{len(EVAL_PROMPTS)}  finetuned={finetuned_totals[p]}/{len(EVAL_PROMPTS)}")
    print("\n" + "=" * 80)
    print(report)


@app.local_entrypoint()
def main():
    run_comparison.remote()