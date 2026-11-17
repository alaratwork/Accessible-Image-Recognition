# architecture — how every file and function connects

this document goes one level more granular than the
[who-calls-what diagram](../diagrams/architecture.png), which shows which controller
calls which function. this shows function-to-function connections in detail, and explains the reasoning behind every
mvc boundary decision - not just what goes where, but why.

## the mvc split, and why each boundary sits where it does

**model** = anything that transforms or represents data, with no opinion
about who's asking or how the result gets shown. a model function could be
called from a test script, a controller, or a different project entirely,
and it would behave identically.

**view** = anything whose whole job is presentation. a view function
receives data that's already decided, and only decides how to show it -
never what the data should be.

**controller** = anything that decides what to do next. a controller reads
a photo path off argv, decides to call segmentation, then decides to call
prediction on the result, then decides to call the view. the decisions
live here; the mechanics live in model and view.

a few boundary calls were genuinely judgment, not obvious, and are worth
stating outright:
- `build_configurable_classifier` and `build_train_val_loaders` (used only
  by the experiment controller) live in `model/classifier.py`, not in
  `controller/experiment_controller.py`, because "what does a variant of
  this network look like" is a model question regardless of who's asking
  for the variant.
- the actual round-robin search algorithm - what to try next, based on
  results so far - lives in the controller, not the model, because
  deciding what to do next is exactly what a controller does. the model
  only answers "here's the accuracy for this one specific configuration."
- `view/speech_view.py` keeps its own tiny `if __name__` self-test block,
  which looks like it breaks the "controllers are the only entry points"
  rule. it doesn't, really - it's testing the view in isolation, the same
  way you'd proofread a script before an actor performs it. no model, no
  controller, no orchestration involved.
- the debug plots (`view/debug_view.py`) are still a view, even though
  their audience is me, the developer, not the end user. they take data
  and decide how to present it - just to a screen for verification,
  instead of an ear for accessibility. same category, different audience.

## model/segmentation.py

| function | what it does | called by |
|---|---|---|
| `load_photo(path)` | reads a photo, returns (color array, grayscale array) | `segment_photo`, `segment_single_item_photo` |
| `binarize_and_dilate(gray)` | otsu threshold + dilate, auto-inverts when the corners say the background is the bright side | `segment_photo`, `segment_single_item_photo` |
| `merge_close_boxes(boxes)` | joins boxes likely split from one item | `find_item_boxes` |
| `sort_reading_order(boxes)` | orders boxes row-by-row, left-to-right | `find_item_boxes` |
| `find_item_boxes(dilated)` | contours -> filtered, merged, sorted boxes | `segment_photo` |
| `find_single_item_box(dilated)` | largest region, grown to absorb nearby fragments - one box, no count filters | `segment_single_item_photo` |
| `crop_and_pad(color, box)` | crops one item, pads to square | `segment_photo`, `segment_single_item_photo` |
| `segment_photo(path)` | **the public entry point for multi-item photos** - runs the whole pipeline above, returns (boxes, crops) | `reference_capture_controller.main`, `prediction_controller.main` |
| `segment_single_item_photo(path)` | **the public entry point for photos known to hold one item** - returns (box, crop), box `None` when the photo was already a tight crop and was only squared up | `reference_crop_controller.crop_reference_photos` |

everything above those last two rows is an internal step. no controller
calls `load_photo` or `binarize_and_dilate` directly - they only ever call
`segment_photo` or `segment_single_item_photo`, which are the model's actual
public interface. this is deliberate: if the internal steps change later
(say, a smarter background subtraction method), no controller code needs to
change at all.

there are two public entry points rather than one because the two callers
genuinely know different things. a query photo's item count is the question
being asked, so `segment_photo` has to stay honest about finding 0, 1 or 5
of them. a reference photo's item count was already answered when the photo
was filed by hand into one item's folder, and `segment_single_item_photo`
spends that knowledge on two filters `find_item_boxes` can't safely drop:
the max-area filter (which rejects an already-tight crop as "background
fill") and the exactly-one-contour requirement (which rejects a single item
whose cap and label split into separate contours). collapsing both callers
back into one function would mean picking one behaviour and being wrong for
the other.

## model/classifier.py

| function | what it does | called by |
|---|---|---|
| `build_item_classifier(n)` | frozen resnet18 + fresh n-class fc layer | `training_controller.main`, `model/prediction.py` |
| `build_configurable_classifier(n, unfrozen)` | same, but with a choice of how many layers stay unfrozen | `experiment_controller.run_one_config` |
| `build_transform()` | standard imagenet preprocessing | `load_item_data`, `build_train_val_loaders`, `model/prediction.py` |
| `load_item_data(path)` | ImageFolder dataloader over the full reference set | `training_controller.main` |
| `build_train_val_loaders(path, batch_size)` | splits reference set into train/val for tuning | `experiment_controller.run_one_config` |
| `train_item_classifier(model, modules, loader)` | the actual training loop | `training_controller.main`, `experiment_controller.run_one_config` |
| `evaluate_classifier(model, loader)` | runs eval mode, returns accuracy | `training_controller.main`, `experiment_controller.run_one_config` |

`train_item_classifier` and `evaluate_classifier` are each called from two
different controllers - training and experiment both need to train and
score a model, they just do it with different data splits and different
configurations. before the mvc restructure, experiment.py had its own
separate copy of this training loop. now there's one copy, used both ways.

## model/prediction.py

| function | what it does | called by |
|---|---|---|
| `load_trained_classifier()` | loads the saved model + class name list from disk | `prediction_controller.main` |
| `predict_from_image(model, names, pil_image)` | runs one image through the model, returns (label, confidence) | `predict_from_path`, `prediction_controller.main` |
| `predict_from_path(model, names, path)` | opens a file, then calls `predict_from_image` | (available if a controller needs file-based prediction; the current prediction controller uses `predict_from_image` directly, since segmentation already hands it an in-memory crop) |

## view/speech_view.py

| function | what it does | called by |
|---|---|---|
| `announce_match(name, confidence)` | speaks one result, confidence-aware wording | `prediction_controller.main` |
| `announce_top_k(results, k)` | speaks several candidates when no single one is confident | (available for future use - not yet wired into a controller, since the current confidence handling in `announce_match` already covers the single-result case) |
| `announce_no_items_found()` | speaks a distinct message when segmentation finds nothing | `prediction_controller.main` |

## view/debug_view.py

| function | what it does | called by |
|---|---|---|
| `plot_detected_items(crops)` | numbered grid, for checking a query-photo segmentation | `prediction_controller.main` |
| `plot_labeled_session(crops, class_order)` | grid labeled by assigned item name, for checking a reference session | `reference_capture_controller.main` |
| `plot_confusion_summary(preds, actuals)` | per-item accuracy bar chart | (available for the final held-out evaluation - not yet wired into a controller, since that evaluation script doesn't exist until real query photos do) |

## controller/reference_capture_controller.py

**what it orchestrates:** one photo of all 9 items, laid out in a fixed
grid → sorted into `data/reference/<item_name>/`.

```
main()
  → model.segmentation.segment_photo(photo_path)
  → view.debug_view.plot_labeled_session(crops, CLASS_ORDER)     [verify before trusting]
  → save_reference_session(crops, CLASS_ORDER, session_number)   [defined in this file - saving/organizing reference data is this controller's own decision, not a reusable model operation another controller would need]
```

run 6 times total (once per planned capture session), changing
`photo_path` and `session_number` each time.

## controller/training_controller.py

**what it orchestrates:** the populated reference folder → a trained,
saved model.

```
main()
  → model.classifier.build_item_classifier(n_classes)
  → model.classifier.load_item_data(training_set_path)
  → model.classifier.train_item_classifier(model, [model.fc], train_loader)
  → model.classifier.evaluate_classifier(model, train_loader)   [sanity check, printed]
  → torch.save(...) + class_names.txt written directly
```

## controller/prediction_controller.py

**what it orchestrates:** the actual task_flow.mermaid experience - one
photo, possibly several items, each one announced.

```
main(photo_path)
  → model.prediction.load_trained_classifier()
  → model.segmentation.segment_photo(photo_path)
  → if nothing detected: view.speech_view.announce_no_items_found()  [stop here]
  → view.debug_view.plot_detected_items(crops)                       [verify before trusting]
  → for each crop:
      → crop_to_pil(crop)                          [defined in this file - converting cv2's BGR array to the PIL RGB format prediction expects is purely a format-adaptation step between two layers, not a reusable model or view operation]
      → model.prediction.predict_from_image(model, class_names, pil_image)
      → view.speech_view.announce_match(label, confidence)
```

this is the file that changed the most during the mvc restructure - the
version before this only accepted one already-cropped image at a time.
since restructuring meant touching this file anyway, it was upgraded to
actually orchestrate the full multi-item photo experience the task flow
diagram always described, instead of a partial slice of it.

## controller/experiment_controller.py

**what it orchestrates:** the round-robin hyperparameter search itself -
the search algorithm lives here, not in the model.

```
main()
  → run_experiment(reference_dir)
      → for each round, for each dimension, for each candidate value:
          → run_one_config(learning_rate, unfrozen, batch_size, reference_dir)
              → model.classifier.build_train_val_loaders(...)
              → model.classifier.build_configurable_classifier(...)
              → model.classifier.train_item_classifier(...)
              → model.classifier.evaluate_classifier(...)
      → carries the best value of each dimension forward into the next sweep
  → save_results(results)   [writes experiment_results.csv]
```

verified during the restructure with a stand-in accuracy function in place
of real training - confirmed the search actually converges to a known
optimum and stays stable, same check already used on P5's version and on
the very first draft of this file.

## controller/single_item_reference_controller.py

**what it orchestrates:** a flat inbox of newly-taken single-item photos →
filed, unmodified, into `data/reference_raw/<item_name>/`.

```
main()
  → sort_single_item_photos(source_folder, dest_folder, CLASS_ORDER)   [defined in this file - filing photos by name is this controller's own decision, not a model operation]
      → parse_item_name(filename, CLASS_ORDER)   [defined in this file]
      → shutil.copy2(...)                        [copies, never moves - source stays intact]
```

this controller calls nothing in `model/` at all, which is the point. it
used to call `segment_photo` and write crops straight into
`data/reference/`, giving the project two competing crop paths and writing
into what is now a write-only output folder. sorting is a filing job;
cropping is an image-processing job. splitting them is what lets the crop
step downstream stay safely re-runnable.

## controller/reference_crop_controller.py

**what it orchestrates:** `data/reference_raw/` (read-only) → the framing
consistency crop → `data/reference/` (write-only).

```
main()
  → guard: data/reference_raw missing or empty → say so and stop   [see below]
  → crop_reference_photos(raw_dir, output_dir)   [defined in this file]
      → for each item folder, for each photo:
          → model.segmentation.segment_single_item_photo(source_path)
          → cv2.imwrite(output_path, crop)       [into data/reference/, never back into the source]
```

the reason reference photos get cropped at all: query photos are always
tightly cropped by segmentation before classification, so a reference photo
framed as a normal camera shot - counter and background visible - would
train the model at a scale prediction never sees.

**the read-only/write-only split is load-bearing, not tidiness.** an earlier
version of this file read and wrote the same `data/reference/` folder,
cropping in place. that made it destructive and not idempotent: a second run
fed already-cropped images back through segmentation, and a tight crop
leaves otsu almost no background to split against, so the "item" found the
second time was a smaller patch inside the real item. every extra run ate
further in. it genuinely destroyed reference photos, one down to 247x247.
with the folders split, the source is read-only by construction, so a rerun
recomputes identical output from identical input instead of compounding on
itself - verified by hashing the whole output tree after two consecutive
runs and confirming it byte-identical, and confirming the source hash
unchanged.

`main` refuses to run silently on a missing or empty source folder. an
earlier run printed *literally nothing* - the source folder didn't exist
under the name expected, every `listdir` entry got skipped as "not a
directory", and the per-item summary then had an empty dict to loop over, so
a total no-op was indistinguishable from a crash. that cost real debugging
time, so both cases now say so out loud.

## controller/evaluation_controller.py

**what it orchestrates:** labeled held-out query photos → the top-1 accuracy
figure, against its 80% target.

```
main()
  → guard: data/query_labeled missing → say so and stop
  → model.prediction.load_trained_classifier()
  → evaluate_labeled_photos(model, class_names, labeled_dir)   [defined in this file]
      → for each labeled photo:
          → parse_expected_items(filename)        [defined in this file - ground truth]
          → model.segmentation.segment_photo(path)
          → for each crop: crop_to_pil → model.prediction.predict_from_image
          → match_predictions(expected, predicted)  [defined in this file - multiset compare]
  → save_results(rows)                            [writes evaluation_results.csv]
  → view.debug_view.plot_confusion_summary(pred_labels, true_labels)
```

this is the file that finally wires up `plot_confusion_summary`, which existed
in the view layer from the start with nothing calling it.

ground truth lives in the filename, not a separate label file, so a photo can
never drift apart from its label: `lemon_1.jpg` is one lemon,
`lemon+orange_1.jpg` is two items, `apricot+lemon+orange_1.jpeg` is three. The
trailing `_<n>` is a counter and is ignored.

**scoring ignores order, deliberately.** `cv2.imread` does not apply a photo's
EXIF rotation, so a photo that looks vertically arranged on screen can be
stored side-on, and `sort_reading_order` works over stored pixels rather than
visual orientation. `IMG_5206` is a real example: three items appear
top-to-bottom as apricot, orange, lemon, and `segment_photo` returns lemon,
orange, apricot. Comparing as multisets rather than sequences is what keeps
that from being scored as three errors instead of three correct answers. The
comparison is a multiset intersection rather than a set one so that a photo
containing two of the same item can't be scored as one.

`main` also prints which classes have no query photograph at all, so the
accuracy figure is never reported without its denominator.

## the data folders, and which direction data flows

| folder | written by | read by | may I edit it by hand? |
|---|---|---|---|
| `data/raw_singles/` | me, off the phone | `single_item_reference_controller` | yes - it's the inbox |
| `data/reference_raw/` | `single_item_reference_controller`, or me by hand | `reference_crop_controller` | yes - but treat as originals, never let a script write here |
| `data/reference/` | `reference_crop_controller` | `training_controller`, `experiment_controller` | **no** - fully regenerable output, deleting it is safe |
| `data/reference_sessions/` | **retired** - see below | `reference_capture_controller` | n/a - the folder is gone |
| `data/query_photos/` | me, off the phone | `prediction_controller` | yes - held out for the final evaluation, never trained on |
| `data/query_labeled/` | me, by naming copies `<item>[+<item>]_<n>.ext` | `evaluation_controller` | yes - the filename IS the ground truth |
| `data/reference_raw_unusable/` | me, quarantining photos too damaged to train on | nothing | yes - kept only so the loss is documented, never read by code |

the one rule that matters: **`data/reference/` is output, not data.** if it
looks wrong, delete the whole folder and re-run the crop controller. every
photo in it can be rebuilt from `data/reference_raw/`. nothing else in the
project is safe to delete that way, and nothing should ever write back into
`data/reference_raw/`.

**why `data/reference_sessions/` was retired.** it held two multi-item photos
for the grid-capture path. those same two photos were later promoted to query
photos - they are `apricot+lemon+orange_1` and `_2` in `data/query_labeled/`,
and they account for 6 of the 10 item instances in the reported accuracy. that
made the folder a silent leakage trap: `reference_capture_controller` reads
`data/reference_sessions/` and writes crops into `data/reference_raw/`, which
trains, so running it would have trained the model on its own evaluation set
and invalidated the accuracy figure without raising a single error. the photos
survive in `data/query_photos/` and `data/query_labeled/`; the controller
stays in the codebase and now fails loudly on a missing file rather than
quietly poisoning the training set.

the general rule behind that: **a photograph is either reference or query,
never both.** the folder a photo lives in is a claim about its role, and when
the role changes the folder has to change with it.

## running any of this

every controller must be run as a module, from the project root, using
dots and no `.py` extension - e.g. `python3 -m controller.training_controller`.

running it as a plain script instead (`python3 controller/training_controller.py`)
looks like it should work, since you're standing in the right directory,
but it isn't the same thing: python puts the *script's own folder*
(`controller/`) on its search path, not the project root you're standing
in, so `model.` and `view.` imports fail with `ModuleNotFoundError: No
module named 'model'` even though the directory you're in is correct.
this bit us for real - it's not a hypothetical edge case.
