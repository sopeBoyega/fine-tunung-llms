# JSX Style Fine-Tuning with QLoRA

This project is a proof of concept for fine-tuning a code LLM to generate React/Next.js components in a specific personal coding style.

It builds a small style-conditioned dataset, fine-tunes `Qwen/Qwen2.5-Coder-1.5B-Instruct` with QLoRA on Modal, and evaluates whether the fine-tuned adapter follows concrete JSX style patterns better than the base model.

## What This Project Does

- Extracts real React/Next.js component examples as style anchors.
- Uses Gemini to generate synthetic JSX examples that follow the same style guide.
- Fine-tunes Qwen2.5-Coder-1.5B-Instruct with 4-bit QLoRA.
- Saves the LoRA adapter to a Modal volume.
- Compares the base model and fine-tuned model on held-out component prompts.
- Scores outputs against measurable style patterns, not just subjective preference.

## Target Style

The fine-tune is designed to encourage patterns such as:

- Tailwind arbitrary values like `text-[13px]`, `h-[39px]`, and `border-[1.21px]`
- typed `Props` objects
- destructured props in component signatures
- sparse comments
- union types for constrained props
- `export default ComponentName` at the bottom of the file
- React/Next.js component code with a compact personal style

## Project Structure

```text
.
├── prep_data.py              # Builds the JSONL fine-tuning dataset
├── train.py                  # Runs QLoRA fine-tuning on Modal
├── eval.py                   # Compares base vs fine-tuned model outputs
├── comparison_report.md      # Generated evaluation report
├── requirements.txt          # Local Python dependency list
└── README.md
```

## Requirements

- Python 3.11+
- Modal account and CLI
- Hugging Face account/token
- Gemini API key
- Access to an A100 GPU through Modal for training and evaluation

Install the local dependency:

```bash
pip install -r requirements.txt
```

The Modal functions install their own runtime dependencies inside Modal images.

## Secrets

Create the required Modal secrets before running the pipeline.

For Gemini dataset generation:

```bash
modal secret create gemini GEMINI_API_KEY=your_gemini_api_key
```

For Hugging Face model access:

```bash
modal secret create huggingface HF_TOKEN=your_huggingface_token
```

## Workflow

### 1. Generate the Dataset

```bash
modal run prep_data.py
```

This creates a dataset at:

```text
/data/jsx_dataset.jsonl
```

inside the Modal volume:

```text
jsx-finetune-vol
```

The dataset combines:

- 5 real style-anchor examples
- up to 20 synthetic examples generated with Gemini

This is intentionally small and should be treated as a portfolio/demo fine-tune, not a large-scale training dataset.

### 2. Train the LoRA Adapter

```bash
modal run --detach train.py
```

Training uses:

- base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- QLoRA with 4-bit quantization
- LoRA rank `r=16`
- LoRA alpha `32`
- 8 epochs
- sequence length `1024`
- Modal A100 GPU

The final adapter is saved to:

```text
/data/outputs/jsx-final-adapters
```

### 3. Evaluate the Fine-Tune

```bash
modal run eval.py
```

The evaluation script runs both the base model and the fine-tuned adapter on held-out JSX prompts, then scores the outputs against style checks such as:

- Tailwind arbitrary values
- typed destructured props
- default export at the bottom
- `React.FC` usage
- union types for constrained props

To download the report from Modal:

```bash
modal volume get jsx-finetune-vol /outputs/jsx_comparison_report.md ./comparison_report.md
```

## Current Results

From the generated comparison report, the fine-tuned adapter improved several style-pattern scores on 6 held-out prompts:

| Pattern | Base model | Fine-tuned |
|---|---:|---:|
| `tailwind_arbitrary_values` | 0/6 | 3/6 |
| `typed_props_destructured` | 0/6 | 2/6 |
| `default_export_at_bottom` | 0/6 | 6/6 |
| `react_fc_typing` | 1/6 | 3/6 |
| `union_type_for_constrained_prop` | 0/6 | 0/6 |

These results suggest the adapter learned some surface-level coding conventions from the style dataset. Because the dataset is small, the project should be understood as a focused experiment in style conditioning rather than a broad capability improvement.

## Optional: Push Adapter to Hugging Face

In `train.py`, set:

```python
HF_PUSH_REPO = "your-username/qwen2.5-coder-1.5b-jsx-style-lora"
```

Then rerun training or push after the adapter is saved.

By default, `HF_PUSH_REPO` is set to `None`, so the adapter stays on the Modal volume.

## Notes

- The dataset generation script can resume from an existing dataset file on the Modal volume.
- Training resumes from the latest saved checkpoint if one exists.
- Evaluation sampling uses `temperature=0.2`, so exact outputs may vary slightly between runs.
- The scoring is heuristic, but it gives a concrete signal for whether the adapter shifted toward the desired style.

## License

No license has been specified yet. Add one before publishing if you want others to reuse or modify the project.
