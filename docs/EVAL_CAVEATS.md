# Evaluation caveats

Two defects found on a 2026 re-read of the code, after all runs were done. Both are left
in place so the code matches the logged Weights & Biases runs; this file is the record.

## 1. BLEU figures are invalid (scoring bug)

`pt_app/trainer/evaluation.py:182-183` computes corpus BLEU like this:

```python
refs_formatted = [[ref] for ref in references]
bleu_score = sacrebleu.corpus_bleu(predictions, refs_formatted)
```

`sacrebleu.corpus_bleu` expects a list of reference *streams*: one list per reference
set, each aligned with `predictions`. For a single reference per sentence the correct
call is `corpus_bleu(predictions, [references])`. The code passes N streams of length 1
instead. sacrebleu zips hypotheses against the streams, which truncates to the first
hypothesis, and the N reference sentences are then treated as N alternative references
for that one hypothesis. The reported "corpus BLEU" is therefore the score of the first
test sentence against every reference in the set.

Reproduction with sacrebleu 2.5.1 (the pinned version): on a 3-sentence toy set the buggy
call reports a system length of one sentence and a score of 100.0; the correct call
reports 47.9. The same pattern is at line 415.

Consequence: the base-vs-fine-tuned BLEU delta of 4.14 → 23.36 (W&B run `20251121_121031`)
and the decoding-sweep BLEU values (27.90 / 29.98 / 20.11, run `20251119_131447`) are not
translation-quality results. ROUGE-L, perplexity and the quality-filter pass rate in the
same runs do not go through this code path and stand.

Fix: replace `[[ref] for ref in references]` with `[references]` and re-score the saved
predictions.

## 2. The judge's test set leaks its training sources

The judge training data is built by `pt_app/eval_model/judge_gen.py`: for each source
sentence, the teacher generates several Portuguese translations at controlled quality
levels, each with a 0 to 10 score, issues and feedback. `split_judge_data.py` then
shuffles those rows and cuts 80/10/10. Because the shuffle is at row level, the variants
of one source sentence land in train, val and test alike.

On the local split used for training and evaluation (`datasets/judge_eval/`, not
committed, 138 test rows): all 138 test rows have a source sentence that also appears in
the training set. The whole set of 1,372 rows covers only 124 distinct source sentences.
The judge saw the source, the reference and several scored translations of every test
item during training. The Cohen's κ / MAE / Pearson r that
`pt_app/eval_model/eval_judge_fast.py` computes against the teacher on that test set
measure memorisation of source-specific patterns, not judging ability, and are not
reported. From my notes (not from a file in this repo): the agreement I measured was at
or below chance.

Fix: group rows by source sentence before splitting, and hold out sources the judge has
never seen. `check_data_overlap.py` only checks file consistency between the final set and
the splits; it does not detect this leak.
