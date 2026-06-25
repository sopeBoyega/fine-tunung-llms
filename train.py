"""
Modal port of notebook 2 — QLoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct.

Run with:
    modal run train.py

Requires prep_data.py to have been run first (writes train/eval to the volume).
Also requires a Modal Secret named "huggingface" with key HF_TOKEN set, e.g.:
    modal secret create huggingface HF_TOKEN=hf_xxxxxxxxxxxx
"""

import modal

app = modal.App("codefinetune-train")

volume = modal.Volume.from_name("finetune-vol", create_if_missing=True)

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

# Push the final adapter straight to your HF account when training finishes.
# Set to None to skip the push and just keep the adapter on the volume.
HF_PUSH_REPO = "Sope006/fine-tuning-llms"  # e.g. "your-username/qwen2.5-coder-1.5b-codefeedback-lora"


@app.function(
    image=image,
    gpu="A100",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3 * 3600,  # 3 hours ceiling — adjust down once you know real runtime
)
def train():
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer
    from datasets import load_from_disk
    from huggingface_hub import login

    login(token=os.environ["HF_TOKEN"])

    LOCAL_OUT = "/tmp/training-output"
    os.makedirs(LOCAL_OUT, exist_ok=True)

    # ── Tokenizer ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 4-bit quantisation (unchanged from your notebook) ─────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print("Quantisation config ready")

    # ── Load model in 4-bit + LoRA ──────────────────────────────────
    print("Loading model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    print(f"VRAM after load: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} ({100 * trainable / total:.2f}% of total)")

    # ── Load datasets ────────────────────────────────────────────
    import datasets as datasets_lib
    print(f"datasets library version: {datasets_lib.__version__}")
    train_ds = load_from_disk(f"{VOLUME_PATH}/data/train")
    eval_ds = load_from_disk(f"{VOLUME_PATH}/data/eval")

    def format_conv(ex):
        return {"text": tokenizer.apply_chat_template(ex["conversations"], tokenize=False, add_generation_prompt=False)}

    train_ds = train_ds.map(format_conv, num_proc=4)
    eval_ds = eval_ds.map(format_conv, num_proc=4)
    print(f"Train: {len(train_ds):,}  Eval: {len(eval_ds):,}")

    # ── Training config — A100 on Modal, so the A100 branch always applies ──
    BATCH = 4
    ACCUM = 4
    SEQ_LEN = 2048
    print(f"batch={BATCH} accum={ACCUM} seq={SEQ_LEN} effective_batch={BATCH * ACCUM}")

    training_args = SFTConfig(
        output_dir=LOCAL_OUT,
        num_train_epochs=3,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=ACCUM,
        per_device_eval_batch_size=BATCH,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        group_by_length=True,
        seed=42,
        packing=False,
        max_length=SEQ_LEN,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # ── Save adapter to volume (replaces the Drive copy step) ──────
    ADAPTER_PATH = f"{VOLUME_PATH}/outputs/final-adapters"
    model.save_pretrained(ADAPTER_PATH)
    tokenizer.save_pretrained(ADAPTER_PATH)
    volume.commit()
    print(f"Adapters saved to volume: {ADAPTER_PATH}")

    for f in os.listdir(ADAPTER_PATH):
        size_mb = os.path.getsize(f"{ADAPTER_PATH}/{f}") / 1e6
        print(f"  {f}: {size_mb:.1f} MB")

    # ── Optional: push straight to Hugging Face Hub ────────────────
    if HF_PUSH_REPO:
        print(f"Pushing adapter to {HF_PUSH_REPO}...")
        model.push_to_hub(HF_PUSH_REPO)
        tokenizer.push_to_hub(HF_PUSH_REPO)
        print("Pushed.")


@app.local_entrypoint()
def main():
    train.remote()