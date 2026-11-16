# Spoken Pantry Identifier
# final project: accessible pantry item identifier
# CONTROLLER layer - crops sorted reference photos from a read-only source
# Does: walks every data/reference_raw/<item_name>/ folder, segments each
#       photo, and writes the tightly-cropped single-item result to the
#       matching data/reference/<item_name>/ folder
# Does not: do any sorting or renaming - the folder structure already tells
#       us which item each photo is, since that sorting was done by hand.
#       this file only fixes framing consistency, nothing about
#       organization. it also never writes to data/reference_raw/, ever.
# Reasoning: query photos always get tightly cropped by segmentation before
#       classification (model/segmentation.py's segment_photo, via
#       prediction_controller.py). if reference photos are framed
#       differently, e.g. a normal camera shot with counter and background
#       visible, training sees a different scale than prediction ever will
#       - the same framing mismatch already observed once, honestly, in the
#       pilot test's low-confidence lemon result. this makes every
#       reference photo go through that identical crop step, so training
#       and prediction see consistently-framed images.
#
#       this replaces an earlier version that cropped in place, reading and
#       writing the same data/reference/ folder. that version was
#       destructive and NOT idempotent: running it a second time re-fed
#       already-cropped images back through segmentation, and a tight crop
#       has almost no background left for otsu to split against, so the
#       "item" it re-found was some smaller interior patch. each extra run
#       ate further into the photo. it genuinely destroyed several
#       previously-good crops, one down to 247x247 from a 4032px original.
#       splitting the folders is what makes reruns safe - the source is
#       read-only by construction, so a second run recomputes identical
#       output from identical input instead of compounding on itself.
#
# run as a module from the project root:
#       python3 -m controller.reference_crop_controller

# imports start
import sys
import os
import cv2
from model.segmentation import segment_single_item_photo
# imports end


# useful functions start

# walks every subfolder of raw_dir, segments every photo in it, and writes
# each cropped result into the matching subfolder of output_dir.
# raw_dir is only ever read from; output_dir is only ever written to.
# uses segment_single_item_photo rather than segment_photo because the
# folder structure already guarantees one item per photo - see that
# function's comment for why that assumption buys so much here. it reports
# each photo as either cropped or passed through untouched, so a photo that
# was already tightly cropped is visible in the output as such instead of
# being silently re-cropped and degraded.
def crop_reference_photos(raw_dir, output_dir):
    processed = {}
    skipped = []

    for item_name in sorted(os.listdir(raw_dir)):
        item_folder = os.path.join(raw_dir, item_name)
        if not os.path.isdir(item_folder):
            continue

        # mirror each source item folder into the output tree, creating it
        # if this is the first run
        output_folder = os.path.join(output_dir, item_name)
        os.makedirs(output_folder, exist_ok=True)

        processed[item_name] = 0
        for filename in sorted(os.listdir(item_folder)):
            if filename.startswith('.'):
                continue

            source_path = os.path.join(item_folder, filename)
            box, crop = segment_single_item_photo(source_path)

            if crop is None:
                print('  {:<24} {:<28} skipped - unreadable'.format(item_name.replace('_', ' '), filename))
                skipped.append(source_path)
                continue

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, crop)
            processed[item_name] += 1
            # the file name and one word of status, rather than two full paths
            # and a pixel box - during a demo the interesting fact is whether
            # the photo needed cropping at all
            print('  {:<24} {:<28} {}'.format(
                item_name.replace('_', ' '), filename,
                'already tight' if box is None else 'cropped'))

    return processed, skipped
# useful functions end


# main function start

# reads data/reference_raw/, writes data/reference/ - safe to run as many
# times as needed, since the source folder is never modified
def main(argv):
    raw_dir = 'data/reference_raw'
    output_dir = 'data/reference'

    # fail loudly if the source folder isn't there. an earlier run of this
    # pipeline printed literally nothing at all - the source folder didn't
    # exist under the name expected, every listdir entry got skipped as
    # "not a directory", and the per-item summary then had an empty dict to
    # loop over, so a total no-op looked exactly like a silent crash. a
    # missing input folder is worth an explicit message, not silence.
    if not os.path.isdir(raw_dir):
        print('  {} does not exist - nothing to prepare'.format(raw_dir))
        print('  put the original photos in {}/<item_name>/ first'.format(raw_dir))
        return

    print()
    print('pantry identifier  ·  preparing reference photos')
    print()
    processed, skipped = crop_reference_photos(raw_dir, output_dir)

    print()
    if not processed:
        print('  {} has no item folders - nothing to prepare'.format(raw_dir))
    total = sum(processed.values())
    print('  {} photos prepared across {} items'.format(total, len(processed)))
    if skipped:
        print()
        print('  {} photo(s) skipped - check these by hand:'.format(len(skipped)))
        for path in skipped:
            print('    {}'.format(path))
    return

if __name__ == "__main__":
    main(sys.argv)
# main function end
