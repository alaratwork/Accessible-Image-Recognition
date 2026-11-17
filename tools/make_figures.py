# Spoken Pantry Identifier
# final project: accessible pantry item identifier
# Does: build the three photo-based figures the deck needs and cannot get from
#       a terminal capture - the whole training set at once, a detection lined
#       up against the photo it came from, and the shelf failure case
# Does not: invent or retouch anything. every crop is produced by the real
#       segmentation call and every label by the real classifier, at the
#       moment this runs
# Reasoning: presentation/slide_visuals/ shows what the program SAYS. these
#       show what it SEES, which is the more convincing thing for two of the
#       claims in the talk: that 13 photographs is genuinely very little, and
#       that the failures are failures of localisation rather than naming.
#
# run from the project root:  python3 tools/make_figures.py

# imports start
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
import cv2
import numpy as np
from PIL import Image
from model.segmentation import segment_photo, load_photo
from model.prediction import load_trained_classifier, predict_from_image
# imports end


# useful functions start

OUT_DIR = 'presentation/figures'
TILE = 240
FONT, SCALE, THICK = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
GREEN, AMBER, WHITE = (120, 220, 140), (80, 200, 250), (255, 255, 255)


# scales a crop to a square tile and writes a caption band under it, so every
# figure lines up on a grid no matter what shape the original crop was
def tile(crop, top_text='', bottom_text='', bottom_colour=WHITE):
    t = cv2.resize(crop, (TILE, TILE))
    band = np.full((34, TILE, 3), 24, np.uint8)
    out = np.vstack([t, band])
    if top_text:
        cv2.putText(out, top_text[:26], (6, 20), FONT, SCALE, GREEN, THICK, cv2.LINE_AA)
    if bottom_text:
        cv2.putText(out, bottom_text[:26], (6, TILE + 23), FONT, SCALE, bottom_colour, THICK, cv2.LINE_AA)
    return out


# lays tiles out in a grid, padding the last row so the image stays rectangular
def grid(tiles, columns):
    rows = []
    for i in range(0, len(tiles), columns):
        row = tiles[i:i + columns]
        while len(row) < columns:
            row.append(np.full_like(tiles[0], 24))
        rows.append(np.hstack(row))
    return np.vstack(rows)


# figure 1 - every reference photo the model trained on, in one image. the
# point of this one is how few there are
def dataset_figure():
    tiles = []
    for item in sorted(os.listdir('data/reference')):
        folder = os.path.join('data/reference', item)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.startswith('.'):
                continue
            crop = cv2.imread(os.path.join(folder, name))
            tiles.append(tile(crop, item[:24]))
    out = grid(tiles, 5)
    cv2.imwrite(os.path.join(OUT_DIR, 'dataset_13_photos.png'), out)
    return len(tiles), out.shape


# figure 2 - the photo with numbered boxes drawn on it, beside the crops those
# boxes produced and what each was called. this is the "it lines up" evidence
def lineup_figure(model, names):
    path = 'data/query_labeled/apricot+lemon+orange_1.jpeg'
    colour, _ = load_photo(path)
    boxes, crops = segment_photo(path)

    marked = colour.copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(marked, (x, y), (x + w, y + h), AMBER, 16)
        cv2.putText(marked, str(i), (x + 26, y + 190), FONT, 6.0, AMBER, 16, cv2.LINE_AA)
    scale = 820.0 / marked.shape[1]
    marked = cv2.resize(marked, (820, int(marked.shape[0] * scale)))

    strip = []
    for i, crop in enumerate(crops):
        label, confidence = predict_from_image(model, names, Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        strip.append(tile(crop, '{}  {}'.format(i, label), '{:.2f} confidence'.format(confidence)))
    strip = np.vstack(strip)

    height = max(marked.shape[0], strip.shape[0])
    canvas = np.full((height, marked.shape[1] + strip.shape[1] + 24, 3), 255, np.uint8)
    canvas[:marked.shape[0], :marked.shape[1]] = marked
    canvas[:strip.shape[0], marked.shape[1] + 24:] = strip
    cv2.imwrite(os.path.join(OUT_DIR, 'detection_lineup.png'), canvas)
    return len(crops), canvas.shape


# figure 3 - what the four cluttered shelf photos actually produce. every tile
# here is a piece of shelving, which is the honest point
def failure_figure(model, names):
    tiles = []
    for name in ['IMG_5244.jpeg', 'IMG_5245.jpeg', 'IMG_5246.jpeg', 'IMG_5247.jpeg']:
        boxes, crops = segment_photo(os.path.join('data/query_photos', name))
        for crop in crops:
            label, confidence = predict_from_image(model, names, Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            tiles.append(tile(crop, name.replace('.jpeg', ''), '{} {:.2f}'.format(label[:14], confidence), AMBER))
    out = grid(tiles, 5)
    cv2.imwrite(os.path.join(OUT_DIR, 'shelf_failure.png'), out)
    return len(tiles), out.shape
# useful functions end


# main function start

def main():
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

    model, names = load_trained_classifier()

    count, shape = dataset_figure()
    print('  dataset_13_photos.png   {} reference crops    {}x{}'.format(count, shape[1], shape[0]))

    count, shape = lineup_figure(model, names)
    print('  detection_lineup.png    {} items, boxed and named   {}x{}'.format(count, shape[1], shape[0]))

    count, shape = failure_figure(model, names)
    print('  shelf_failure.png       {} artifact crops     {}x{}'.format(count, shape[1], shape[0]))

    print()
    print('wrote 3 figures to {}/'.format(OUT_DIR))
    return

if __name__ == "__main__":
    main()
# main function end
