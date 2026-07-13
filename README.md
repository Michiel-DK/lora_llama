# lora_llama

LoRA fine-tuning of **Llama-3.2-1B-Instruct** for English→Portuguese translation,
with a separate **LLM-as-a-judge** model (Qwen2.5-3B) trained to score translation
quality. Runs on Apple Silicon (MPS) locally and NVIDIA GPUs (CUDA / Vast.ai) in
the cloud. All configuration is in `params.py`.

## Results

Fine-tuned on OpenSubtitles EN→PT, evaluated on a held-out OPUS test set
(beam-search decoding, numbers from a logged W&B run):

| Model       | BLEU  | ROUGE-L | Perplexity | Filter pass rate |
|-------------|-------|---------|------------|------------------|
| Baseline    | 4.14  | 0.5528  | 1.86       | 86.7%            |
| Fine-tuned  | 23.36 | 0.6869  | 1.57       | 100.0%           |
| Improvement | +19.2 (+464%) | +0.134 | +0.29 | +13.3 pts |

Decoding strategy on the fine-tuned model (separate sweep):

| Strategy    | BLEU  | ROUGE-L | Perplexity | Filter pass rate |
|-------------|-------|---------|------------|------------------|
| greedy      | 27.90 | 0.5166  | 1.99       | 85.8%            |
| beam search | 29.98 | 0.5954  | 1.70       | 94.5%            |
| sampling    | 20.11 | 0.5103  | 1.85       | 85.8%            |

Scores vary run-to-run with sample size and seed. Reproduce with
`compare_baseline_vs_finetuned.py` and `compare_generation_strategies.py`.

## Examples

Fine-tuned model output vs. the reference translation:

| English | Model output | Reference |
|---------|--------------|-----------|
| You seem to like fruit. | Você parece gostar de frutas. | Você parece gostar de frutas. |
| We've got to find Tom first. | Temos que encontrar Tom primeiro. | Temos que encontrar o Tom primeiro. |
| Tom is always in a good mood. | Tom sempre está em um bom humor. | Tom está sempre de bom humor. |
| Tom sings in the school chorus. | Tom canta na coralagem escolar. | Tom canta no coral da escola. |
| This vacuum cleaner is noisy. | Esta máquina de limpeza de pólvora é ruim. | Este aspirador de pó é barulhento. |

The model sometimes appends an explanatory note or picks a wrong sense on rare
vocabulary; the quality filter (`pt_app/trainer/quality_filter.py`) strips notes,
language-mixing, and repetition before scoring.

## Layout

| Path | What |
|------|------|
| `params.py` | Config: model, LoRA, optimizer, generation strategies |
| `train.py` | Entry point: build data, sanity-check, train, evaluate |
| `pt_app/data/` | Dataset loading, preprocessing, chat-template formatting |
| `pt_app/trainer/` | LoRA training loop (`trainer_pt.py`), evaluation (`evaluation.py`), quality filter, stopping criteria |
| `pt_app/eval_model/` | Judge model: data gen, training (`_mps` local / `_cuda` cloud), eval |
| `run_inference.py` | CLI to translate with a trained adapter |
| `compare_baseline_vs_finetuned.py` | Base vs. fine-tuned comparison |
| `compare_generation_strategies.py` | greedy / beam / sampling sweep |
| `docs/` | Detailed guides (training, inference, MPS vs CUDA, Vast.ai setup) |

## Usage

```bash
pip install -r requirements.txt          # add requirements_cuda.txt on NVIDIA GPUs
cp .env.example .env                      # then fill in your tokens

# Train the translation adapter (config in params.py)
python train.py

# Translate with a trained adapter
python run_inference.py --adapter ./adapters/<run_name> --interactive
python run_inference.py --adapter ./adapters/<run_name> --prompt "Hello, how are you?"

# Compare base vs. fine-tuned
python compare_baseline_vs_finetuned.py --adapter ./adapters/<run_name>
```

## Judge model

Rather than relying only on BLEU/ROUGE, a Qwen2.5-3B model is fine-tuned to predict
a translation quality score, using Groq-generated reference scores as targets.
Agreement with the reference is measured with Cohen's κ and MAE. See
[`docs/JUDGE_TRAINING_USAGE.md`](docs/JUDGE_TRAINING_USAGE.md) and
[`docs/VAST_AI_SETUP.md`](docs/VAST_AI_SETUP.md).

## Notes

- Requires a Hugging Face token with access to the Llama-3.2 and Qwen models.
- `adapters/`, `outputs/`, `datasets/`, and `wandb/` are gitignored — weights and
  data stay local.
- Experiment tracking via Weights & Biases / Weave (toggle `USE_WANDB` in `params.py`).
- Fully pinned environment in `requirements-lock.txt`.
