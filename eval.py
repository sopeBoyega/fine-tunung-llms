"""
Base vs fine-tuned comparison — runs both the original Qwen2.5-Coder-1.5B-Instruct
and your LoRA adapter on the same held-out eval prompts, side by side.

Run with:
    modal run eval_compare.py

Requires train.py to have already run and saved the adapter to the volume
at /data/outputs/final-adapters.

Output: a markdown file written to the volume with prompts, both completions,
and basic stats (length, eval-set loss already computed during training).
Pull it down with:
    modal volume get finetune-vol /outputs/comparison_report.md ./comparison_report.md
"""

import modal

app = modal.App("codefinetune-eval")

volume = modal.Volume.from_name("finetune-vol", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.0",
    "transformers==4.46.3",
    "peft==0.13.2",
    "bitsandbytes==0.44.1",
    "accelerate==1.0.1",
    "datasets==2.21.0",
    "huggingface_hub==0.26.2",
)

VOLUME_PATH = "/data"
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = f"{VOLUME_PATH}/outputs/final-adapters"

# Held-out prompts the model has NOT been trained on directly — chosen to span
# a few common coding-assistant task types so the comparison isn't one-note.
EVAL_PROMPTS = [
    "Write a Python function that checks if a string is a palindrome, ignoring case and spaces.",
    "Write a function in JavaScript that debounces another function by a given delay in milliseconds.",
    "Given a list of dictionaries representing students with 'name' and 'grade' keys, write Python code to find the student with the highest grade.",
    "Write a TypeScript interface and a function that validates whether an object matches that interface at runtime.",
    "Explain and fix the bug in this Python code:\n\ndef divide(a, b):\n    return a / b\n\nresult = divide(10, 0)",
    "Write a SQL query to find the second-highest salary from an 'employees' table.",
    "Write a Python decorator that times how long a function takes to run and prints the result.",
    "Convert this for-loop into a list comprehension:\n\nresult = []\nfor x in range(20):\n    if x % 3 == 0:\n        result.append(x * x)",
]


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

    def generate(model, prompt, max_new_tokens=300):
        messages = [
            {"role": "system", "content": "You are an expert programming assistant. Write clean, efficient, well-commented code."},
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

    # ── Run base model first, then free it before loading the adapter ──
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

    # ── Now load base + adapter ─────────────────────────────────────
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

    # ── Build markdown report ───────────────────────────────────────
    lines = [
        "# Base vs Fine-Tuned Comparison",
        "",
        f"Base model: `{MODEL_ID}`",
        f"Adapter: LoRA, r=8, alpha=16, trained 1 epoch on filtered Magicoder + CodeFeedback",
        "",
        "Note: completions are generated with temperature=0.2 (sampling), so re-running",
        "will produce slightly different text even from the same model. Judge the overall",
        "pattern across prompts, not any single completion.",
        "",
        "---",
        "",
    ]

    for i, prompt in enumerate(EVAL_PROMPTS):
        lines.append(f"## Prompt {i+1}")
        lines.append("")
        lines.append(f"**Prompt:** {prompt}")
        lines.append("")
        lines.append("**Base model:**")
        lines.append("```")
        lines.append(base_outputs[i])
        lines.append("```")
        lines.append("")
        lines.append("**Fine-tuned model:**")
        lines.append("```")
        lines.append(finetuned_outputs[i])
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    report = "\n".join(lines)

    report_path = f"{VOLUME_PATH}/outputs/comparison_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    volume.commit()

    print(f"\nReport written to volume at {report_path}")
    print("Pull it down with:")
    print(f"  modal volume get finetune-vol /outputs/comparison_report.md ./comparison_report.md")

    # Also print it directly to logs so you can read it without pulling the file
    print("\n" + "=" * 80)
    print(report)


@app.local_entrypoint()
def main():
    run_comparison.remote()