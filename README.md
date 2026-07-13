# lora_llama

LoRA fine-tuning of **Llama-3.2-1B-Instruct** for English→Portuguese translation,
plus a separate **LLM-as-a-judge** model (Qwen2.5-3B) trained to score translation
quality — distilled from a larger Groq-hosted teacher. Runs on Apple Silicon (MPS)
locally and NVIDIA GPUs (CUDA / Vast.ai) in the cloud. All configuration is in
`params.py`.

## Results

Fine-tuned on OpenSubtitles EN→PT, evaluated on a held-out OPUS test set
(beam-search decoding, numbers from a logged W&B run):

| Model       | BLEU  | ROUGE-L | Perplexity | Filter pass rate |
|-------------|-------|---------|------------|------------------|
| Baseline    | 4.14  | 0.5528  | 1.86       | 86.7%            |
| Fine-tuned  | 23.36 | 0.6869  | 1.57       | 100.0%           |
| Improvement | +19.2 (+464%) | +0.134 | +0.29 | +13.3 pts |

Decoding strategy for the fine-tuned **translation** model (separate sweep; beam
search wins on BLEU and ROUGE-L):

| Strategy    | BLEU  | ROUGE-L | Perplexity | Filter pass rate |
|-------------|-------|---------|------------|------------------|
| greedy      | 27.90 | 0.5166  | 1.99       | 85.8%            |
| beam search | 29.98 | 0.5954  | 1.70       | 94.5%            |
| sampling    | 20.11 | 0.5103  | 1.85       | 85.8%            |

Scores vary run-to-run with sample size and seed. Reproduce with
`compare_baseline_vs_finetuned.py` and `compare_generation_strategies.py`.

## Examples

Base Llama-3.2-1B vs. the fine-tuned model on the same inputs (from the comparison
run above). The base model often picks a wrong sense or appends an explanatory
note; the fine-tuned model is concise and idiomatic:

| English | Base model | Fine-tuned | Reference |
|---------|------------|-----------|-----------|
| This vacuum cleaner is noisy. | Esta máquina de limpeza de **pólvora** é ruim. | A máquina de limpeza é **barulhosa**. | Este aspirador de pó é barulhento. |
| Tom sings in the school chorus. | Tom canta na coralagem escolar. *Nota: "chorus"...* | O Tom cantava no coro da escola. | Tom canta no coral da escola. |
| Let's play some video games to kill time. | Vamos jogar alguns jogos de vídeo... *Essa tradução mantém...* | Vamos jogar videogames para passar o tempo. | Vamos jogar video game para matar o tempo. |
| We've got to find Tom first. | Temos que encontrar Tom primeiro. | Temos de encontrar **o** Tom primeiro. | Temos que encontrar o Tom primeiro. |

The base model's appended notes and verbosity are exactly what tanks its BLEU
(4.14 above); the quality filter (`pt_app/trainer/quality_filter.py`) also strips
such notes, language-mixing, and repetition before scoring.

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

## Judge model (LLM-as-a-judge, distilled from a teacher)

Rather than relying only on BLEU/ROUGE, a Qwen2.5-3B model is fine-tuned to score
translation quality. The training data is **distilled from a larger teacher**: a
Groq-hosted model takes each source/reference pair and generates several Portuguese
translations at controlled quality levels, each labelled with a score, the specific
issues, and feedback (`pt_app/eval_model/judge_gen.py`). The small judge learns to
reproduce those scores, and its agreement with the teacher is measured with
Cohen's κ and MAE (`pt_app/eval_model/eval_judge_fast.py`).

Trained on Apple Silicon (`judge_train_mps.py`) or a cloud NVIDIA GPU
(`judge_train_cuda.py`). See [`docs/JUDGE_TRAINING_USAGE.md`](docs/JUDGE_TRAINING_USAGE.md)
and [`docs/VAST_AI_SETUP.md`](docs/VAST_AI_SETUP.md).

## Notes

- Requires a Hugging Face token with access to the Llama-3.2 and Qwen models.
- `adapters/`, `outputs/`, `datasets/`, and `wandb/` are gitignored — weights and
  data stay local.
- Experiment tracking via Weights & Biases / Weave (toggle `USE_WANDB` in `params.py`).
- Fully pinned environment in `requirements-lock.txt`.
