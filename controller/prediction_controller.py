# Spoken Pantry Identifier
# final project: accessible pantry item identifier
# CONTROLLER layer - the real end-to-end flow a user actually experiences
# Does: loads the trained classifier, segments one raw photo - or every
#       photo in a folder - into the items it contains, classifies each
#       crop, and speaks a result for each one: the actual
#       task_flow.mermaid experience, start to finish
# Does not: contain any segmentation math, network code, or speech-
#       synthesis code itself - it only decides what to call, in what
#       order, and hands results between model and view
# Reasoning: this replaces an earlier version that only accepted one
#       pre-cropped image at a time - upgraded while restructuring into
#       mvc, since a controller whose whole job is orchestration should
#       actually orchestrate the full multi-item experience, not a
#       partial slice of it.
#
# run as a module from the project root:
#       python3 -m controller.prediction_controller <photo>
# as a module, not a script - running it as python3 controller/prediction_
# controller.py puts controller/ on sys.path instead of the project root, so
# every "from model...."/"from view...." import in this file fails outright.

# imports start
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')  
import sys
import cv2
from PIL import Image
from model.segmentation import segment_photo
from model.prediction import load_trained_classifier, predict_from_image
from view.speech_view import create_speech_engine, announce_match, announce_no_items_found
from view.debug_view import plot_detected_items
# imports end


# useful functions start

# opencv crops are BGR numpy arrays; the model's transform expects a PIL
# RGB image - this is the one conversion point where segmentation's output
# format meets prediction's expected input format
def crop_to_pil(crop):
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


# collects the photos a path refers to - either the one file named, or every
# photo inside the directory named, sorted. a directory is worth supporting
# because it matches how someone actually uses this: they photograph several
# things, drop the photos into a folder, and want all of them read out. one
# photo at a time is the special case of that, not the other way round.
def collect_photo_paths(path):
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        extensions = ('.jpg', '.jpeg', '.png')
        return [os.path.join(path, name) for name in sorted(os.listdir(path))
                if not name.startswith('.') and name.lower().endswith(extensions)]
    return []


# segments one photo and announces every item found in it. kept separate from
# main so the multi-photo loop reads as "do this to each photo" rather than
# nesting two loops in the entry point. takes the already-created engine
# rather than making its own, for the one-engine-per-process reason above.
def announce_photo(photo_path, model, class_names, engine, show_plot=True):
    boxes, crops = segment_photo(photo_path)
    print()
    print('  reading   {}'.format(os.path.basename(photo_path)))

    if not crops:
        print('  found     nothing')
        print()
        announce_no_items_found(engine)
        return 0

    print('  found     {} item{}'.format(len(crops), '' if len(crops) == 1 else 's'))
    print()

    if show_plot:
        plot_detected_items(crops)

    for i, crop in enumerate(crops, 1):
        pil_image = crop_to_pil(crop)
        label, confidence = predict_from_image(model, class_names, pil_image)
        # the status word carries the same threshold the speech uses, so what is
        # printed and what is said can never disagree
        status = 'confident' if confidence >= 0.6 else 'not sure'
        print('  {}  {:<22} {:<10} {:.0%}'.format(i, label.replace('_', ' '), status, confidence))
        announce_match(engine, label, confidence)
    return len(crops)
# useful functions end


# main function start

# takes one raw photo OR a directory of them, and announces a result for every
# item detected in each - the actual task_flow.mermaid experience.
# creates exactly one speech engine for the whole run and reuses it across
# every announcement, no matter how many photos are processed - see
# speech_view.create_speech_engine's comment for why that matters.
def main(argv):
    if len(argv) < 2:
        print('  usage:  python3 -m controller.prediction_controller <photo or folder>')
        return

    target = argv[1]
    photo_paths = collect_photo_paths(target)
    if not photo_paths:
        print('  nothing to read at {}'.format(target))
        print('  expected a photo, or a folder containing photos')
        return

    print()
    print('pantry identifier')
    model, class_names = load_trained_classifier()
    engine = create_speech_engine()

    # the debug plot is skipped when several photos are queued, because every
    # plot blocks until it's closed by hand - fine for inspecting one photo,
    # but it turns a folder of ten into ten interruptions
    show_plot = len(photo_paths) == 1

    total_items = 0
    for photo_path in photo_paths:
        total_items += announce_photo(photo_path, model, class_names, engine, show_plot)

    if len(photo_paths) > 1:
        print()
        print('  total     {} item{} across {} photos'.format(
            total_items, '' if total_items == 1 else 's', len(photo_paths)))
    return

if __name__ == "__main__":
    main(sys.argv)
# main function end
