# Spoken Pantry Identifier
# final project: accessible pantry item identifier
# CONTROLLER layer - orchestrates one reference-capture session
# Does: reads one photo, calls segmentation to detect+crop every item,
#       shows a labeled plot to verify before trusting it, saves each
#       crop into its class-named folder
# Does not: contain any image-processing logic itself (that's
#       model/segmentation.py) or any plotting logic itself (that's
#       view/debug_view.py) - it only decides what to call and when
# Reasoning: one run per capture session, varying angle and lighting each
#       time. this is the grid capture path the proposal described, and it
#       produced none of the reference set actually used - that was shot as
#       individual single-item photos instead. CLASS_ORDER must match the
#       physical layout of items in every photo exactly, since position
#       maps directly to class name.
#
# run as a module from the project root:
#       python3 -m controller.reference_capture_controller

# imports start
import sys
import os
import cv2
from model.segmentation import segment_photo
from view.debug_view import plot_labeled_session
# imports end


# useful functions start

# saves each detected crop into data/reference_raw/<item_name>/<session>.png -
# position in reading order maps directly to CLASS_ORDER, so items must be
# laid out in the exact same left-to-right, top-to-bottom order every
# session - same discipline prepare_greek_examples.py required for its
# row-to-letter mapping.
# writes to data/reference_raw/, NOT data/reference/, even though what it
# writes is already cropped. data/reference/ has exactly one writer -
# reference_crop_controller.py - and is deleted and rebuilt from source
# whenever anything changes; anything this file put there directly would be
# silently destroyed by the next rebuild. landing in the source folder
# instead means these crops survive, and the crop step recognizes them as
# already tight and passes them through untouched on its way to
# data/reference/. one generated folder, one writer, no exceptions.
def save_reference_session(crops, class_order, session_number, base_folder='data/reference_raw'):
    if len(crops) != len(class_order):
        print('*** warning: detected {} items but CLASS_ORDER has {} names - '
              'check the layout/photo before trusting this session'.format(len(crops), len(class_order)))

    saved = 0
    for i, crop in enumerate(crops):
        if i >= len(class_order):
            break
        folder = os.path.join(base_folder, class_order[i])
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, '{}.png'.format(session_number)), crop)
        saved += 1
    print('session {}: saved {} items into {}/<item_name>/'.format(session_number, saved, base_folder))
# useful functions end


# main function start

# CLASS_ORDER must match the physical layout of items in every reference
# photo, exactly, every session - left-to-right, top-to-bottom.
# 9 items, not the 10 the proposal named: the tenth was dropped because no
# reference photographs of it were ever captured. see the matching note in
# single_item_reference_controller.py - the two lists must stay in step,
# since both are used to recognize the same set of item names.
#
# one caveat specific to THIS file: because this controller maps grid
# position to item name by index, removing a name also shifts every item
# after it by one position. any grid photo laid out for the old 10-item
# order would now be labeled wrong from 'apple_cider_vinegar' onward. that
# is safe here only because the reference set actually used was captured as
# individual single-item photos, not grids - this controller produced none
# of the reference data in use. re-photograph the grid against this list
# before running it again.
CLASS_ORDER = [
    'ketchup_bottle', 'mayonnaise_bottle', 'salt', 'cinnamon', 'lemon',
    'apricot', 'orange', 'apple_cider_vinegar', 'avocado',
]

# point photo_path at the grid photo for the session being processed.
# data/reference_sessions/ was retired - the two photos it held became
# query photos, and this controller writes into data/reference_raw/, which
# trains, so keeping them there would have trained on the evaluation set.
# see ARCHITECTURE.md. put a new grid photo somewhere of its own first.
def main(argv):
    photo_path = 'data/reference_sessions/reference_session_2.jpg'
    session_number = 2

    boxes, crops = segment_photo(photo_path)
    print('detected {} items (expecting {})'.format(len(crops), len(CLASS_ORDER)))

    plot_labeled_session(crops, CLASS_ORDER)
    save_reference_session(crops, CLASS_ORDER, session_number)
    return

if __name__ == "__main__":
    main(sys.argv)
# main function end
