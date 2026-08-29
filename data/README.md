# Data documentation

## `model_family_panel.csv`

The panel contains 57 public base-model variants from Meta, DeepSeek, Ai2,
Alibaba, and Google DeepMind. Each row records the released configuration and
links to the official paper, model card, or release report supporting it.

The analysis uses `eligible_for_dn=true` only when the source reports a
model-specific processed-token count. Family-level dataset or mixture sizes
remain in the panel for transparency but do not enter the parameter--token
regression. Mixture-of-experts generations are excluded from that regression
because active and total parameters capture different costs.

Important fields:

- `total_params_b`, `active_params_b`: billions of parameters;
- `training_tokens_t`: trillions of processed tokens or, where explicitly
  marked, a family dataset/mixture size;
- `eligible_for_dn`: whether the token count is model-specific;
- `primary_source`, `source_note`: row-level provenance and interpretation.

## `llama3_isoflop_experiments.csv`

The file contains 133 Llama 3 IsoFLOP observations at ten compute budgets. It
is the `source == "llama_3"` subset of the Open Athena standardized IsoFLOP
dataset:

- Dataset: <https://huggingface.co/datasets/open-athena/isoflop-experiments>
- Llama 3 report: <https://arxiv.org/abs/2407.21783>

Open Athena reports that these points were digitized from Meta's published
scaling figure. The package treats them as the public experimental surface and
does not claim that they cover the released 8B, 70B, or 405B endpoints.

## Units

- `params`: parameter count, not billions;
- `tokens`: processed token count, not trillions;
- `budget`: training FLOPs;
- `loss`: reported validation loss.

The two CSV files are analysis inputs. Running the package requires no network
access and does not alter them. `SHA256SUMS` records their verified hashes.
The code license at the repository root does not supersede terms attached to
the cited source datasets and publications.
