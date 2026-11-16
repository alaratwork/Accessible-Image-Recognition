# Spoken Pantry Identifier
# final project: accessible pantry item identifier
# CONTROLLER layer - the final held-out evaluation the proposal promised
# Does: walks every labeled query photo, segments it, classifies each crop,
#       compares the predictions against the ground truth encoded in the
#       filename, and reports overall and per-item top-1 accuracy
# Does not: train anything, touch the reference set, or speak - this is a
#       measurement run for the report, not the user-facing experience.
#       prediction_controller.py is still the file a real user drives
# Reasoning: the proposal commits to a specific success criterion - top-1
#       accuracy on held-out query photos, target 80% - and nothing in the
#       project measured it. model/prediction.py could classify one image
#       and view/debug_view.py could plot a per-item accuracy bar chart,
#       but no controller had ever connected the two against ground truth.
#       this file is that connection, and it is deliberately the only place
#       ground truth is interpreted.
#
#       ground truth comes from the filename rather than a separate label
#       file, so a photo can never drift apart from its own label:
#           lemon_1.jpg                     -> one item, a lemon
#           lemon+orange_1.jpg              -> two items, a lemon and an orange
#           apricot+lemon+orange_1.jpeg     -> three items
#       the trailing _<n> is just a counter to allow several photos of the
#       same item set, and is ignored.
#
#       scoring compares the predicted labels against the expected labels as
#       MULTISETS, ignoring order. that is not laziness, it is required for
#       correctness: cv2.imread does not apply a photo's EXIF rotation, so a
#       photo that looks vertically arranged on screen can be stored
#       side-on, and segmentation's reading order is computed over the
#       stored pixels rather than the visual orientation. IMG_5206 is a real
#       example - three items appear top-to-bottom as apricot, orange,
#       lemon, but segment_photo returns them as lemon, orange, apricot.
#       order-sensitive scoring would have called that three errors when
#       every item was in fact found and correctly identified.
#
# run as a module from the project root:
#       python3 -m controller.evaluation_controller

# imports start
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # must be set before importing torch
os.environ.setdefault('OMP_NUM_THREADS', '1')          # torch + pyttsx3 in one process segfaults without this
import sys
import csv
import cv2
from PIL import Image
from model.segmentation import segment_photo
from model.prediction import load_trained_classifier, predict_from_image
from view.debug_view import plot_confusion_summary
# imports end


# useful functions start

# pulls the expected item names out of a filename, in the order written.
# 'lemon+orange_1.jpg' -> ['lemon', 'orange'].  the trailing _<n> counter is
# dropped only when it really is a counter, so an item name that itself ends
# in an underscored word can't be truncated by accident
def parse_expected_items(filename):
    stem = os.path.splitext(filename)[0]
    parts = stem.split('_')
    if parts[-1].isdigit():
        parts = parts[:-1]
    return '_'.join(parts).split('+')


# same BGR-array-to-PIL-RGB conversion prediction_controller.py needs, for
# the same reason - segmentation hands back cv2 arrays, the model's
# transform expects PIL
def crop_to_pil(crop):
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


# scores one photo's predictions against its expected items as multisets, so
# order doesn't matter. returns (matched_pairs, missed_items, spurious_preds)
# where matched_pairs are the expected items that were correctly predicted.
# a multiset intersection is the right comparison here rather than a set one
# because a photo can legitimately contain two of the same item, and set
# logic would silently score that as one
def match_predictions(expected, predicted):
    remaining = list(predicted)
    matched, missed = [], []
    for item in expected:
        if item in remaining:
            remaining.remove(item)
            matched.append(item)
        else:
            missed.append(item)
    return matched, missed, remaining


# runs every labeled photo through segmentation + classification and scores
# it. returns the per-photo rows plus the flat (predicted, actual) pairs
# view/debug_view.py's plot_confusion_summary expects
def evaluate_labeled_photos(model, class_names, labeled_dir):
    rows = []
    pred_labels, true_labels = [], []

    for filename in sorted(os.listdir(labeled_dir)):
        if filename.startswith('.'):
            continue

        expected = parse_expected_items(filename)
        unknown = [item for item in expected if item not in class_names]
        if unknown:
            print('  {:<32} skipped - unknown item name {}'.format(filename, unknown))
            continue

        path = os.path.join(labeled_dir, filename)
        boxes, crops = segment_photo(path)

        predicted, confidences = [], []
        for crop in crops:
            label, confidence = predict_from_image(model, class_names, crop_to_pil(crop))
            predicted.append(label)
            confidences.append(confidence)

        matched, missed, spurious = match_predictions(expected, predicted)

        # every expected item becomes one scored instance. a missed item is
        # recorded against its true name with a '(not found)' prediction, so
        # the per-item chart shows which items get lost rather than just a
        # lower overall number
        for item in matched:
            pred_labels.append(item)
            true_labels.append(item)
        for item in missed:
            pred_labels.append('(not found)')
            true_labels.append(item)

        rows.append({
            'photo': filename,
            'expected_items': '+'.join(expected),
            'detected_count': len(crops),
            'expected_count': len(expected),
            'predicted_items': '+'.join(predicted) if predicted else '(none)',
            'correct': len(matched),
            'missed': '+'.join(missed) if missed else '',
            'spurious': '+'.join(spurious) if spurious else '',
            'confidences': ' '.join('{:.3f}'.format(c) for c in confidences),
        })

        print('  {:<32} {:<24} {} of {}'.format(
            filename, ' '.join(p.replace('_', ' ') for p in predicted) if predicted else 'nothing',
            len(matched), len(expected)))

    return rows, pred_labels, true_labels


# writes the per-photo detail to csv for the report
def save_results(rows, path='evaluation_results.csv'):
    if not rows:
        return
    fieldnames = ['photo', 'expected_items', 'expected_count', 'detected_count',
                  'predicted_items', 'correct', 'missed', 'spurious', 'confidences']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
# useful functions end


# main function start

# scores data/query_labeled/ - photos shot separately from the reference set
# and never trained on. the proposal's target is 80% top-1.
def main(argv):
    labeled_dir = 'data/query_labeled'

    # same lesson as the other controllers: an absent input folder should say
    # so, not produce an empty report that looks like a result
    if not os.path.isdir(labeled_dir):
        print('*** {} does not exist - nothing to evaluate.'.format(labeled_dir))
        print('    name query photos <item>[+<item>...]_<n>.jpg and put them there.')
        return

    print()
    print('pantry identifier  ·  held-out evaluation')
    print()
    model, class_names = load_trained_classifier()
    rows, pred_labels, true_labels = evaluate_labeled_photos(model, class_names, labeled_dir)

    if not true_labels:
        print()
        print('*** no scorable photos found in {}.'.format(labeled_dir))
        return

    total = len(true_labels)
    correct = sum(1 for p, t in zip(pred_labels, true_labels) if p == t)
    accuracy = 100. * correct / total

    print()
    print('  accuracy  {} of {} correct   {:.0f}%'.format(correct, total, accuracy))
    print('  target    80%   {}'.format('met' if accuracy >= 80.0 else 'not met'))
    print()
    print('  per item')
    for item in sorted(set(true_labels)):
        item_total = sum(1 for t in true_labels if t == item)
        item_correct = sum(1 for p, t in zip(pred_labels, true_labels) if t == item and p == t)
        print('    {:<22} {} of {}'.format(item.replace('_', ' '), item_correct, item_total))

    # how many classes never got tested at all - the honest denominator
    untested = [c for c in class_names if c not in set(true_labels)]
    if untested:
        print()
        print('  not measured   {} of {} classes have no query photograph:'.format(
            len(untested), len(class_names)))
        for name in untested:
            print('    {}'.format(name.replace('_', ' ')))

    save_results(rows)
    plot_confusion_summary(pred_labels, true_labels, all_classes=class_names)
    print()
    print('  saved     evaluation_results.csv, prediction_summary.png')
    return

if __name__ == "__main__":
    main(sys.argv)
# main function end
