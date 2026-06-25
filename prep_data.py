"""
Modal port of notebook 1, Cell 7 — dataset download, cleaning, formatting.

Run with:
    modal run prep_data.py

This downloads Magicoder + CodeFeedback, applies your filters, and saves
train/eval splits to a Modal Volume (replaces the Google Drive save).
"""

import modal

app = modal.App("codefinetune-prep")

volume = modal.Volume.from_name("finetune-vol", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "datasets>=2.18.0",
    "huggingface_hub",
)

VOLUME_PATH = "/data"


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=1800,  # 30 min — dataset download + filtering can take a while
)
def prep_data():
    from datasets import load_dataset, concatenate_datasets
    import re

    # ── Step 1: Load and convert to unified format ────────────────
    def convert_magicoder(ex):
        return {
            "conversations": [
                {"role": "system", "content": "You are an expert programming assistant. Write clean, efficient, well-commented code."},
                {"role": "user", "content": ex["problem"]},
                {"role": "assistant", "content": ex["solution"]},
            ]
        }

    def convert_codefeedback(ex):
        return {
            "conversations": [
                {"role": "system", "content": "You are an expert programming assistant. Write clean, efficient, well-commented code."},
                {"role": "user", "content": ex["query"]},
                {"role": "assistant", "content": ex["answer"]},
            ]
        }

    # ── Step 2: Quality filters (unchanged from your notebook) ────
    def is_python_or_js(ex):
        text = " ".join(t["content"] for t in ex["conversations"])
        return bool(re.search(r"```python|```javascript|```typescript|```ts|```py|```js|\bdef \w+\(|\bimport \w+", text, re.I))

    def quality_filter(ex):
        asst = [t for t in ex["conversations"] if t["role"] == "assistant"]
        if not asst:
            return False
        last = asst[-1]["content"]
        if len(last) < 100:
            return False  # too short
        total = sum(len(t["content"]) for t in ex["conversations"])
        if total > 8000:
            return False  # too long -> OOM risk
        if "```" not in last:
            return False  # no code block
        return True

    # ── Step 3: Load, convert, filter ──────────────────────────────
    print("Loading Magicoder...")
    mag = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    mag = mag.map(convert_magicoder, remove_columns=mag.column_names, num_proc=4)
    mag = mag.filter(is_python_or_js).filter(quality_filter)
    print(f"Magicoder after filter: {len(mag):,}")

    print("Loading CodeFeedback...")
    cfb = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
    cfb = cfb.filter(lambda ex: ex["lang"] in ("python", "javascript", "typescript"))
    cfb = cfb.map(convert_codefeedback, remove_columns=cfb.column_names, num_proc=4)
    cfb = cfb.filter(quality_filter)
    print(f"CodeFeedback after filter: {len(cfb):,}")

    # ── Step 4: Combine, shuffle, split ────────────────────────────
    combined = concatenate_datasets([mag, cfb]).shuffle(seed=42)
    test_size = min(500, int(len(combined) * 0.02))
    split = combined.train_test_split(test_size=test_size, seed=42)

    split["train"].save_to_disk(f"{VOLUME_PATH}/data/train")
    split["test"].save_to_disk(f"{VOLUME_PATH}/data/eval")
    volume.commit()  # persist to the volume — required after writes

    print(f"Saved: Train={len(split['train']):,}  Eval={len(split['test']):,}")
    print(f"Data written to volume at {VOLUME_PATH}/data/")


@app.local_entrypoint()
def main():
    prep_data.remote()
