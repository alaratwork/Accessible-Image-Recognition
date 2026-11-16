# Spoken Pantry Identifier
# final project: accessible pantry item identifier
# CONTROLLER layer - sorts individual single-item reference photos by name
# Does: for every photo in a flat source folder, works out which item it is
#       from its filename and copies it, unmodified, into the matching
#       data/reference_raw/<item_name>/ folder
# Does not: crop, segment, or otherwise alter a single pixel - and does not
#       handle multi-item grid photos, which are still
#       reference_capture_controller.py's job. this file exists for a
#       different actual capture style: one photo per item, not one photo
#       of all 9 items together
# Reasoning: the capture style actually used ended up being individual
#       single-item photos (2 per item) rather than the originally planned
#       10-item grid per session.
#
#       this used to segment and crop each photo on its way in, writing
#       crops straight to data/reference/. that put a second, duplicate
#       cropping path in the project, competing with the one in
#       controller/reference_crop_controller.py, and it wrote into what is
#       now a write-only output folder. the pipeline has one crop step now,
#       and this file feeds it rather than racing it:
#           this file:                     flat photos -> data/reference_raw/
#           reference_crop_controller.py:  data/reference_raw/ -> data/reference/
#       sorting is a filing job and cropping is an image-processing job.
#       keeping them separate is what lets the crop step stay re-runnable,
#       since its source folder then holds untouched originals rather than
#       something already cropped once on the way through.
#
# run as a module from the project root:
#       python3 -m controller.single_item_reference_controller

# imports start
import sys
import os
import shutil
# imports end


# useful functions start

# matches a filename like 'ketchup_bottle_1.jpg' or 'lemon_2.jpg' against
# the known item names, returning the item name it belongs to or None.
# checks longest names first so 'ketchup_bottle_1.jpg' can't match a
# shorter name that happens to be a prefix of it
def parse_item_name(filename, class_order):
    stem = os.path.splitext(filename)[0]
    for name in sorted(class_order, key=len, reverse=True):
        if stem.startswith(name):
            return name
    return None


# copies every photo in source_folder into dest_folder/<item_name>/,
# keeping the original filename and file contents exactly as they are.
# copies rather than moves so the source folder still holds the originals
# afterwards - the same read-only-source discipline the crop step relies on
def sort_single_item_photos(source_folder, dest_folder, class_order):
    counters = {}
    for filename in sorted(os.listdir(source_folder)):
        if filename.startswith('.'):
            continue

        item_name = parse_item_name(filename, class_order)
        if item_name is None:
            print("*** skipping {} - doesn't match any known item name".format(filename))
            continue

        out_dir = os.path.join(dest_folder, item_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        shutil.copy2(os.path.join(source_folder, filename), out_path)

        counters[item_name] = counters.get(item_name, 0) + 1
        print('{} -> {}'.format(filename, out_path))

    return counters
# useful functions end


# main function start

# same CLASS_ORDER as reference_capture_controller.py - used here only to
# recognize valid filenames, not for any position-based sorting.
# 9 items, not the 10 the proposal named: the tenth was dropped because no
# reference photographs of it were ever captured, and a name in this list
# with no photographs behind it is worse than an absent one - it silently
# widens the classifier's output layer by one class that can never be
# predicted correctly, and makes every reported class count off by one.
CLASS_ORDER = [
    'ketchup_bottle', 'mayonnaise_bottle', 'salt', 'cinnamon', 'lemon',
    'apricot', 'orange', 'apple_cider_vinegar', 'avocado',
]

# expects data/raw_singles/<item_name>_<n>.jpg for every photo, e.g.
# ketchup_bottle_1.jpg, ketchup_bottle_2.jpg, lemon_1.jpg, lemon_2.jpg -
# a flat inbox folder of newly-taken photos, straight off the phone
def main(argv):
    source_folder = 'data/raw_singles'
    dest_folder = 'data/reference_raw'

    # same lesson as reference_crop_controller.py: an absent input folder
    # used to make this print nothing at all, which is indistinguishable
    # from a crash. say so instead.
    if not os.path.isdir(source_folder):
        print('*** {} does not exist - nothing to sort.'.format(source_folder))
        print('    put newly-taken photos there as <item_name>_<n>.jpg first.')
        return

    counters = sort_single_item_photos(source_folder, dest_folder, CLASS_ORDER)

    print()
    for name in CLASS_ORDER:
        count = counters.get(name, 0)
        flag = '' if count > 0 else '  *** NONE FOUND'
        print('{}: {} photo(s){}'.format(name, count, flag))
    return

if __name__ == "__main__":
    main(sys.argv)
# main function end
