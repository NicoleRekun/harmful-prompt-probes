# Can a model's own activations flag harmful prompts better than a text classifier?

A small interpretability and safety experiment. I take real user prompts that people sent to a chatbot, run them through an open-weights instruction-tuned model, and train a **linear probe** on the model's hidden states at every layer to predict whether the prompt is toxic. I compare that against a **TF-IDF text classifier** that never looks inside a model.

**Status:** pipeline built and tested on synthetic data. First full run on ToxicChat pending. Results will be added below.

## Why this matters

Content filters that only read the text can be dodged by rephrasing, role-play and jailbreak templates. If a model *internally represents* "this request is harmful" even when the wording is disguised, a cheap linear probe on its activations could be a stronger, harder-to-evade safety signal. This project tests that on a small model, and pays particular attention to **jailbreak attempts**, the cases where text-only filters tend to fail.

## Setup

| | |
|---|---|
| Data | [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat) (`toxicchat0124`): real user prompts to the Vicuna demo, human-labelled for toxicity and jailbreaking |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` (any Hugging Face causal LM works via `--model`) |
| Features | Hidden state at the last token after the chat template opens the assistant turn, i.e. the moment the model decides how to respond. Mean pooling available via `--pooling mean` |
| Probe | Standardise + logistic regression (class-balanced), one per layer |
| Baseline | TF-IDF (word 1–2 grams) + logistic regression on the same split |
| Metrics | AUROC, average precision (the class is ~7% positive), F1, and recall on jailbreak prompts |
| Control | Probe retrained on shuffled labels (5 runs). Should land near AUROC 0.5. If not, the probe is learning something other than toxicity |

## Run it

Fastest: open `notebooks/run_on_colab.ipynb` in Google Colab, choose a T4 GPU runtime, and run all cells (about 10 minutes).

Locally:

```bash
pip install -r requirements.txt
python scripts/run_experiment.py --model Qwen/Qwen2.5-0.5B-Instruct
pytest -q   # tests run without a GPU or model download
```

Outputs go to `results/`: `metrics.json` and `auroc_by_layer.png`. Activations are cached in `cache/`, so re-running the probes with different settings doesn't need the model again.

## Results

_To be filled in after the first run:_

- Best probe layer and AUROC vs. TF-IDF baseline
- AUROC-by-layer plot
- Jailbreak recall: probe vs. baseline
- Shuffled-label control
- Error analysis: what kinds of prompts each method misses

## Limitations

- One small model (0.5B parameters). Larger models may represent harmfulness differently.
- ToxicChat labels "toxic" broadly, so this is not a pure "dangerous request" label.
- A linear probe shows that information is *readable* from activations, not that the model *uses* it.
- Jailbreak positives in the test set are few, so jailbreak recall has wide uncertainty.

## Repo layout

```
src/misuse_probe/   data loading, activation extraction (+ cache), probes, baseline
scripts/            run_experiment.py: end-to-end run
tests/              unit tests on synthetic data
notebooks/          Colab runner
```

Uses public research data only.
