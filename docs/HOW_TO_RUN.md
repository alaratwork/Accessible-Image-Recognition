# How to run

> **Does:** get someone from a fresh copy of this folder to a spoken
> identification, and to the reported accuracy number, in that order.
>
> **Does not:** explain the design or the results - [ARCHITECTURE.md](ARCHITECTURE.md)
> covers how the code fits together, the [README](../README.md#results) covers what the numbers mean.
>
> **Reasoning:** two things in this project fail in ways that look like broken code
> rather than user error - running a controller as a script instead of a
> module, and a matplotlib window that blocks before the speech happens.
> both are called out below rather than left to be discovered.

Every command below runs from the project root.

## 1. Requirements

Python 3, and the packages in `requirements.txt`:

```bash
pip3 install -r requirements.txt
```

On macOS with Homebrew Python that becomes:

```bash
pip3 install -r requirements.txt --break-system-packages
```

`pyttsx3` drives the speech. On macOS it uses the built-in voice and needs no
setup. **Turn your volume up** — the spoken result is the actual deliverable of
this project, not the printed text.

## 2. The one thing a user actually does

Point it at a photo, or at a folder of photos:

```bash
python3 -m controller.prediction_controller data/query_labeled/apricot+lemon+orange_1.jpeg
```

```bash
python3 -m controller.prediction_controller data/query_photos
```

The first opens a window showing each detected item, then speaks a result for
each. **Close the window to hear the speech** — see the gotcha below. The second
reads every photo in the folder in one pass and skips the window.

## 3. Reproducing the reported accuracy

```bash
python3 -m controller.evaluation_controller
```

Prints top-1 accuracy over `data/query_labeled/`, per-item accuracy, and which
classes have no query photograph at all. Writes `evaluation_results.csv` and
`prediction_summary.png`. Expect **10/10 = 100.0%**, target met — across 3 of
the 9 classes, which the run states itself.

## 4. Rebuilding from source photographs

`data/reference/` and `item_classifier.pth` are generated, and may be absent
from a size-trimmed copy of this project. Rebuild both from
`data/reference_raw/` in about a minute:

```bash
python3 -m controller.reference_crop_controller
```

```bash
python3 -m controller.training_controller
```

```bash
python3 -m controller.evaluation_controller
```

Both rebuild byte-identically — the crop step is idempotent by construction and
training pins `torch.manual_seed(0)`.

## 5. Every entry point

| command | what it does |
|---|---|
| `python3 -m controller.single_item_reference_controller` | files `data/raw_singles/<item>_<n>.jpg` into `data/reference_raw/<item>/`. Copies only, never crops. |
| `python3 -m controller.reference_capture_controller` | the multi-item grid capture path. Unused for the reported results — the reference set was shot as single items. |
| `python3 -m controller.reference_crop_controller` | `data/reference_raw/` → `data/reference/`. Safe to re-run any number of times. |
| `python3 -m controller.training_controller` | trains, saves `item_classifier.pth` + `class_names.txt`. |
| `python3 -m controller.prediction_controller <photo\|folder>` | segment → classify → speak. The user-facing flow. |
| `python3 -m controller.evaluation_controller` | scores `data/query_labeled/`, reports top-1 accuracy. |
| `python3 -m controller.experiment_controller` | hyperparameter sweep. Runs, but on a dataset this small its differences are noise, so it is not used for any reported result. |
| `python3 -m view.speech_view` | speaks three sample announcements. Quickest way to confirm audio works. |

## Gotchas

**Run controllers as modules, not scripts.** `python3 -m controller.training_controller`
works; `python3 controller/training_controller.py` fails with
`ModuleNotFoundError: No module named 'model'`. Python puts the *script's*
folder on the path rather than the directory you are standing in, so every
`model.` and `view.` import breaks. This is the single most likely thing to go
wrong.

**The plot window blocks, and speech comes after it.** `prediction_controller`
shows the detected crops before announcing anything, so nothing is spoken until
you close that window. It is not hung. To skip the window entirely:

```bash
MPLBACKEND=Agg python3 -m controller.prediction_controller data/query_photos
```

**If the process aborts with an OpenMP error**, PyTorch and `pyttsx3` are
fighting over threads. Every controller already sets `OMP_NUM_THREADS=1` and
`KMP_DUPLICATE_LIB_OK=TRUE` at import time; if you invoke the model layer
directly from your own script, set them before importing torch.

**`data/reference/` is output, not data.** If it looks wrong, delete it and
re-run the crop controller. Everything in it rebuilds from
`data/reference_raw/`, which nothing writes to.
