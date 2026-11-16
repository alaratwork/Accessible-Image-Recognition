# Spoken Pantry Identifier
# final project: accessible pantry item identifier
# MODEL layer - loads a trained classifier and predicts on one image
# Does: loads the saved model and class names from disk, runs one image
#       through it in eval mode, returns a predicted label and a
#       confidence number
# Does not: print, speak, save anything, or know whether the caller is a
#       human testing things or the real prediction controller
# Reasoning: same eval-mode + argmax pattern as predict_examples.py
#       (cs5330 project 5). predict_from_image takes an already-loaded
#       PIL image, not just a file path, so the prediction controller can
#       feed a crop straight from model/segmentation.py without writing
#       it to disk first.

# imports start
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # must be set before importing torch
import torch
import torch.nn.functional as F
from PIL import Image
from model.classifier import build_item_classifier, build_transform
# imports end


# useful functions start

# loads a trained classifier plus the class name list saved during training
def load_trained_classifier(model_path='item_classifier.pth', names_path='class_names.txt'):
    with open(names_path) as f:
        class_names = [line.strip() for line in f if line.strip()]

    model = build_item_classifier(len(class_names))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, class_names


# runs one segmented query image through the classifier, returns the
# predicted class name and its confidence (softmax probability of the top class)
def predict_from_path(model, class_names, image_path):
    image = Image.open(image_path).convert('RGB')
    return predict_from_image(model, class_names, image)


# same as predict_from_path, but takes an already-loaded PIL image instead
# of a file path - lets the prediction controller feed a crop straight from
# model/segmentation.py without writing it to disk first
def predict_from_image(model, class_names, pil_image):
    transform = build_transform()
    tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        log_probs = F.log_softmax(model(tensor), dim=1)
        probs = log_probs.exp()
        confidence, predicted_idx = probs.max(dim=1)

    return class_names[predicted_idx.item()], confidence.item()
# useful functions end
