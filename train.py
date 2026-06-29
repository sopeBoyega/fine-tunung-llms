"""
QLoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct on the JSX/React style dataset
built by prep_jsx_data.py.

Run with:
    modal run --detach train_jsx.py

Requires prep_jsx_data.py to have already run (writes jsx_dataset.jsonl to
the jsx-finetune-vol volume). Also requires a Modal Secret named
"huggingface" with key HF_TOKEN set:
    modal secret create huggingface HF_TOKEN=hf_xxxxxxxxxxxx

NOTE ON DATASET SIZE: this dataset is small (~25 examples: 5 real anchors +
20 synthetic). That's enough to demonstrate the style-conditioning technique
and nudge surface conventions, but it is NOT a large-scale training set.
Expect the model to pick up consistent surface patterns (Tailwind arbitrary
values, prop-destructuring style, sparse comments) rather than deep
generalization. This is intentionally framed as a proof-of-concept in the
model card, not a claim of broad capability improvement.
"""

import modal

app = modal.App("jsx-style-finetune-train")

volume = modal.Volume.from_name("jsx-finetune-vol", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.0",
    "transformers==4.46.3",
    "peft==0.13.2",
    "bitsandbytes==0.44.1",
    "trl==0.11.4",
    "accelerate==1.0.1",
    "datasets==2.21.0",
    "huggingface_hub==0.26.2",
    "rich",
)

VOLUME_PATH = "/data"
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DATASET_PATH = f"{VOLUME_PATH}/jsx_dataset.jsonl"

# Push the final adapter straight to your HF account when training finishes.
# Set to None to skip the push and just keep the adapter on the volume.
HF_PUSH_REPO = None  # e.g. "your-username/qwen2.5-coder-1.5b-jsx-style-lora"


@app.function(
    image=image,
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=2 * 3600,  # small dataset -> this should finish in well under an hour
)
def train():
    import os
    import json
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset
    from huggingface_hub import login

    login(token=os.environ["HF_TOKEN"])

    LOCAL_OUT = f"{VOLUME_PATH}/outputs/jsx-training-output"
    os.makedirs(LOCAL_OUT, exist_ok=True)

    # ── Tokenizer ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 4-bit quantisation (same as the code fine-tune) ───────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    print(f"VRAM after load: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Lower rank than the code fine-tune (r=8) is fine here too, but bumping
    # slightly to r=16 gives the adapter a bit more room to learn the style
    # patterns from a small dataset without just memorizing the 5 anchors.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}% of total)")

    # ── Load the JSONL dataset and split ────────────────────────────
    print(f"Loading dataset from {DATASET_PATH}...")
    raw_examples = []
    with open(DATASET_PATH) as f:
        for line in f:
            raw_examples.append(json.loads(line))
    print(f"Loaded {len(raw_examples)} total examples.")

    real_count = sum(1 for ex in raw_examples if ex.get("source") == "real_anchor")
    synth_count = len(raw_examples) - real_count
    print(f"  Real anchors: {real_count}  Synthetic: {synth_count}")

    # Small dataset -> hold out a handful for eval rather than a percentage
    # split, so eval isn't left with 0-1 examples.
    EVAL_HOLDOUT = min(4, max(1, len(raw_examples) // 8))
    import random
    random.seed(42)
    shuffled = raw_examples[:]
    random.shuffle(shuffled)
    eval_examples = shuffled[:EVAL_HOLDOUT]
    train_examples = shuffled[EVAL_HOLDOUT:]
    print(f"Train: {len(train_examples)}  Eval: {len(eval_examples)}")

    def format_conv(ex):
        return {"text": tokenizer.apply_chat_template(ex["conversations"], tokenize=False, add_generation_prompt=False)}

    train_ds = Dataset.from_list([format_conv(ex) for ex in train_examples])
    eval_ds = Dataset.from_list([format_conv(ex) for ex in eval_examples])

    # ── Training config ─────────────────────────────────────────────
    # Small dataset -> more epochs needed to actually learn the pattern,
    # since each epoch is only ~20 training steps. Batch sized for A100
    # headroom (see the earlier conversation re: underutilized GPU memory
    # on the code fine-tune run).
    BATCH = 4
    ACCUM = 1
    SEQ_LEN = 1024  # component files are short, no need for 2048 here
    EPOCHS = 8       # small dataset needs more passes to actually shift style

    print(f"batch={BATCH} accum={ACCUM} seq={SEQ_LEN} epochs={EPOCHS}")

    training_args = SFTConfig(
        output_dir=LOCAL_OUT,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=ACCUM,
        per_device_eval_batch_size=BATCH,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=2,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=42,
        packing=False,
        max_seq_length=SEQ_LEN,
        dataset_text_field="text",
    )

    class VolumeCommitCallback(TrainerCallback):
        """Commits the Modal volume every time a checkpoint is saved, so
        progress survives even if the connection/container dies mid-run."""
        def on_save(self, args, state, control, **kwargs):
            volume.commit()
            print(f"[checkpoint] step {state.global_step}: committed to volume")
            return control

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[VolumeCommitCallback()],
    )

    print("Starting training...")
    # Resume from the latest checkpoint on the volume if one exists.
    resume_checkpoint = None
    if os.path.isdir(LOCAL_OUT):
        checkpoints = [d for d in os.listdir(LOCAL_OUT) if d.startswith("checkpoint-")]
        if checkpoints:
            latest = max(checkpoints, key=lambda d: int(d.split("-")[1]))
            resume_checkpoint = os.path.join(LOCAL_OUT, latest)
            print(f"Found existing checkpoint, resuming from: {resume_checkpoint}")

    if resume_checkpoint:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()
    print("Training complete!")

    # ── Save adapter to volume ──────────────────────────────────────
    ADAPTER_PATH = f"{VOLUME_PATH}/outputs/jsx-final-adapters"
    model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)
    volume.commit()
    print(f"Adapters saved to volume: {ADAPTER_PATH}")

    for f in os.listdir(ADAPTER_PATH):
        size_mb = os.path.getsize(f"{ADAPTER_PATH}/{f}") / 1e6
        print(f"  {f}: {size_mb:.1f} MB")

    if HF_PUSH_REPO:
        print(f"Pushing adapter to {HF_PUSH_REPO}...")
        model.push_to_hub(HF_PUSH_REPO)
        tokenizer.push_to_hub(HF_PUSH_REPO)
        print("Pushed.")


@app.local_entrypoint()
def main():
    train.remote()