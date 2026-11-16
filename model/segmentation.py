# Spoken Pantry Identifier
# final project: accessible pantry item identifier
# MODEL layer - segments pantry items out of a color photo
# Does: reads a photo, thresholds and dilates it, finds item regions,
#       merges any split contours, sorts them in reading order, crops
#       each one to a square color image. offers this two ways - one for
#       photos whose item count is unknown (segment_photo, for query
#       photos) and one for photos already known to hold exactly one item
#       (segment_single_item_photo, for reference photos)
# Does not: save files, plot anything, decide WHEN to run (controller's
#       job), or decide how to announce a result (view's job)
# Reasoning: adapted from prepare_my_digits.py and prepare_greek_examples.py
#       (cs5330 project 5), which itself mirrors project 3's threshold ->
#       morphology -> region-segmentation pipeline. still works best against
#       a plain, contrasting background - the same "contrasting surface"
#       instruction already built into task_flow.mermaid - but no longer
#       assumes that background is the dark one: binarize_and_dilate reads
#       the corners and flips the threshold's polarity when the surface
#       turns out to be the bright side, which a white counter makes it.

# imports start
import cv2
import numpy as np
# imports end


# useful functions start

# reads a color photo - returns both the color image (for cropping/saving,
# since the classifier needs real color/label info) and a grayscale version
# (for thresholding only).
# 23 of the 30 photos in this project carry an EXIF orientation tag, because a
# phone writes landscape pixels plus a "rotate this 90 degrees" instruction
# rather than rotating the pixels themselves. cv2.imread applies that tag by
# default - verified against cv2.IMREAD_IGNORE_ORIENTATION, which returns the
# un-rotated 5712x4284 where a plain read returns the intended 4284x5712 - so
# the frame here is already oriented the way Preview and Finder show it, and
# reading order is computed over the same layout a person sees. worth stating
# explicitly, because it is version-dependent behaviour that older OpenCV
# (before 3.4.1) did not have.
def load_photo(image_path):
    color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if color is None:
        raise FileNotFoundError('could not read image at {}, check the path'.format(image_path))
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    return color, gray


# thresholds against a dark background, then dilates to close small gaps -
# same otsu approach as the digit/greek-letter pipelines, just against a
# dark background instead of white paper.
# otsu picks WHERE to split light from dark, but it has no idea which side
# of that split is the item and which is the surface. on a dark background
# the item is the bright side and plain THRESH_BINARY is correct. on a
# white background (marble, laminate counter) the polarity silently flips:
# the surface becomes one huge bright region, find_item_boxes throws it out
# for exceeding max_area_fraction, and the photo comes back with 0 items or
# a few meaningless fragments. found this for real on a white marble photo.
# the four corners of a centered item photo are almost certainly background,
# so if their mean brightness sits on the bright side of otsu's own cut, the
# polarity is backwards and the mask needs inverting before dilating.
def binarize_and_dilate(gray, dilate_size=15, corner_fraction=0.05):
    threshold_value, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    height, width = gray.shape[:2]
    corner_h = max(int(height * corner_fraction), 1)
    corner_w = max(int(width * corner_fraction), 1)
    corners = [
        gray[:corner_h, :corner_w],
        gray[:corner_h, width - corner_w:],
        gray[height - corner_h:, :corner_w],
        gray[height - corner_h:, width - corner_w:],
    ]
    corner_brightness = np.mean([patch.mean() for patch in corners])

    if corner_brightness > threshold_value:
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((dilate_size, dilate_size), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return dilated


# merges boxes likely belonging to one item split into two contours (e.g. a
# curved can's label can catch light differently top vs. bottom, splitting
# it into two blobs) - same logic as prepare_my_digits.py's merge_close_boxes.
# gap_threshold_ratio lowered from an initial 0.5 to 0.2 after testing
# against a real photo: 0.5 demanded gaps larger than 3 separate items
# actually had between them in a normal, reasonably compact layout (real
# measured gaps were ~350-450px against a ~1200px median height, well
# under the old 611px threshold), causing 3 distinct items to incorrectly
# merge into 1. genuine same-object splits (the case this function exists
# for) are typically much smaller absolute gaps than that.
def merge_close_boxes(boxes, gap_threshold_ratio=0.2):
    if not boxes:
        return boxes

    boxes = sorted(boxes, key=lambda b: b[0])
    heights = sorted(b[3] for b in boxes)
    median_height = heights[len(heights) // 2]
    gap_threshold = median_height * gap_threshold_ratio

    merged = [list(boxes[0])]
    for b in boxes[1:]:
        last = merged[-1]
        last_right = last[0] + last[2]
        gap = b[0] - last_right
        vertical_overlap = min(last[1] + last[3], b[1] + b[3]) - max(last[1], b[1])
        if gap < gap_threshold and vertical_overlap > 0:
            new_x = min(last[0], b[0])
            new_y = min(last[1], b[1])
            new_right = max(last_right, b[0] + b[2])
            new_bottom = max(last[1] + last[3], b[1] + b[3])
            merged[-1] = [new_x, new_y, new_right - new_x, new_bottom - new_y]
        else:
            merged.append(list(b))
    return [tuple(m) for m in merged]


# sorts boxes in reading order (row by row, left to right within a row) -
# same logic as prepare_greek_examples.py, in case items are laid out in a
# grid rather than a single line
def sort_reading_order(boxes, row_threshold_ratio=0.6):
    if not boxes:
        return boxes

    boxes_by_y = sorted(boxes, key=lambda b: b[1] + b[3] / 2)
    heights = sorted(b[3] for b in boxes)
    median_height = heights[len(heights) // 2]
    row_threshold = median_height * row_threshold_ratio

    rows = []
    current_row = [boxes_by_y[0]]
    current_y = boxes_by_y[0][1] + boxes_by_y[0][3] / 2
    for b in boxes_by_y[1:]:
        cy = b[1] + b[3] / 2
        if abs(cy - current_y) <= row_threshold:
            current_row.append(b)
        else:
            rows.append(sorted(current_row, key=lambda bb: bb[0]))
            current_row = [b]
        current_y = cy
    rows.append(sorted(current_row, key=lambda bb: bb[0]))

    ordered = []
    for row in rows:
        ordered.extend(row)
    return ordered


# finds bounding boxes of connected item regions - filters noise (too small)
# AND background-fill errors (too large, e.g. if the threshold caught most
# of the frame instead of just the items - a real 3d-object failure mode
# that flat ink-on-paper photos didn't have to worry about). returns boxes
# already merged and sorted into reading order.
def find_item_boxes(dilated, min_area_fraction=0.01, max_area_fraction=0.6):
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = dilated.shape[0] * dilated.shape[1]
    min_area = image_area * min_area_fraction
    max_area = image_area * max_area_fraction

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if min_area <= area <= max_area:
            boxes.append((x, y, w, h))

    boxes = merge_close_boxes(boxes)
    boxes = sort_reading_order(boxes)
    return boxes


# crops one item from the COLOR image (not grayscale) with margin - pads
# to square so the later resize to 224x224 doesn't distort the item's
# shape. pads with black, not white - this project assumes a dark
# background throughout (task_flow.mermaid's "contrasting surface" step),
# so black padding blends with the real background instead of introducing
# an artificial white border the original photo never had. the
# prepare_my_digits.py/prepare_greek_examples.py version used white
# because that domain was genuinely ink-on-white-paper; this domain isn't.
def crop_and_pad(color, box, margin=30):
    x, y, w, h = box
    x0 = max(x - margin, 0)
    y0 = max(y - margin, 0)
    x1 = min(x + w + margin, color.shape[1])
    y1 = min(y + h + margin, color.shape[0])
    crop = color[y0:y1, x0:x1]

    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    padded = np.full((side, side, 3), 0, dtype=np.uint8)
    y_off = (side - ch) // 2
    x_off = (side - cw) // 2
    padded[y_off:y_off + ch, x_off:x_off + cw] = crop
    return padded


# convenience function combining the above steps: one photo in, a list of
# (box, color_crop) pairs out, already in reading order. controllers call
# this one function rather than wiring the individual steps together
# themselves every time.
def segment_photo(image_path):
    color, gray = load_photo(image_path)
    dilated = binarize_and_dilate(gray)
    boxes = find_item_boxes(dilated)
    crops = [crop_and_pad(color, box) for box in boxes]
    return boxes, crops


# finds ONE item box in a photo already known to contain exactly one item.
# this is a different problem from find_item_boxes, which has to stay honest
# about how many items it sees because query photos really can hold several.
# reference photos can't - they were sorted by hand into one folder per
# item, so "how many items are in here" is already answered as exactly one,
# and the only question left is where it is.
# that extra knowledge is what lets this drop find_item_boxes' two filters:
#   - no max_area filter. that filter exists to catch a threshold that
#     swallowed the whole frame, but a reference photo that is ALREADY a
#     tight crop legitimately fills 65-92% of its frame, and find_item_boxes
#     was throwing every one of those out as background-fill, reporting
#     0 items on a photo whose item was perfectly visible.
#   - no "exactly one contour" requirement. a single item routinely breaks
#     into several contours - a bottle's cap, label and body catch light
#     differently, and one real avocado photo fragmented into 5 regions.
#     merge_close_boxes only joins horizontal neighbours, so it can't
#     rejoin a vertical split.
# instead: take the largest region, then repeatedly absorb any other region
# lying within merge_gap_ratio of the box built so far. growing outward from
# the biggest region keeps distant specks (glare on a marble counter, a
# shadow at the frame edge) from dragging the box across the whole photo,
# which a blind union of every contour would do.
def find_single_item_box(dilated, min_area_fraction=0.005, merge_gap_ratio=0.35):
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = dilated.shape[0] * dilated.shape[1]
    min_area = image_area * min_area_fraction

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            boxes.append((x, y, w, h))
    if not boxes:
        return None

    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    x, y, w, h = boxes[0]
    left, top, right, bottom = x, y, x + w, y + h

    absorbed = True
    while absorbed:
        absorbed = False
        tolerance = merge_gap_ratio * max(right - left, bottom - top)
        for bx, by, bw, bh in boxes[1:]:
            if bx > right + tolerance or bx + bw < left - tolerance:
                continue
            if by > bottom + tolerance or by + bh < top - tolerance:
                continue
            new_box = (min(left, bx), min(top, by), max(right, bx + bw), max(bottom, by + bh))
            if new_box != (left, top, right, bottom):
                left, top, right, bottom = new_box
                absorbed = True

    return (left, top, right - left, bottom - top)


# segments a photo known to hold exactly one item, and returns (box, crop).
# box comes back None when the item already fills the frame, meaning the
# photo is effectively a finished crop already - in that case the image is
# only squared up, never re-cropped. that distinction is the whole point:
# re-cropping an already-tight crop is exactly the destructive operation
# that damaged several reference photos before controller/reference_crop_
# controller.py was split into read-only source and write-only output. a
# tight crop leaves otsu almost no background to split against, so the
# "item" it finds the second time is some smaller patch inside the real
# item, and the photo shrinks every run. detecting that case and passing the
# image through untouched is what makes a rerun safe even for photos that
# were already cropped once.
def segment_single_item_photo(image_path, fills_frame_fraction=0.85):
    color, gray = load_photo(image_path)
    dilated = binarize_and_dilate(gray)
    box = find_single_item_box(dilated)

    height, width = gray.shape[:2]
    full_box = (0, 0, width, height)
    if box is None:
        return None, crop_and_pad(color, full_box, margin=0)

    if (box[2] * box[3]) / float(width * height) > fills_frame_fraction:
        return None, crop_and_pad(color, full_box, margin=0)

    return box, crop_and_pad(color, box)
# useful functions end
