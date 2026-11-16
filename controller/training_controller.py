# Spoken Pantry Identifier
# final project: accessible pantry item identifier
# CONTROLLER layer - orchestrates training the classifier
# Does: pins the random seed, builds a fresh classifier, loads all reference
#       data, trains it, checks its own training-set accuracy, saves the
#       model + class names to disk
# Does not: contain any network architecture or training-loop math itself
#       (that's model/classifier.py) - it only decides what to call, with
#       which arguments, and what to do with the result
# Reasoning: passes early_stop_threshold=99.0 to train_item_classifier,
#       matching greek_transfer.py's real, proven value - stop once
#       training is basically perfect rather than always running the full
#       epoch count. the experiment controller deliberately does NOT pass
#       this, so every swept configuration trains for the same duration.
#       it seeds torch for the same reason experiment_controller already
#       does - see the comment in main. without it, the saved model differs
#       every run, and no confidence number in the report reproduces.

# imports start
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # must be set before importing torch
import sys
import torch
from model.classifier import build_item_classifier, load_item_data, train_item_classifier, evaluate_classifier
# imports end


# main function start

# expects data/reference/<item_name>/ subfolders, one per pantry item,
# populated by controller/reference_capture_controller.py
def main(argv):
    # same fixed seed experiment_controller.run_one_config already uses, for
    # the same reason: without it, load_item_data's shuffle and the fresh fc
    # layer's random initialization make every run produce a differently
    # calibrated model. that isn't a cosmetic difference. two runs over the
    # identical 13 reference photos disagreed by 0.22 on the same query crop
    # (a correctly-labeled orange scored 0.59 in one run and 0.37 in the
    # next), which is enough to move it from above the speech view's 0.6
    # confidence threshold to below it - a different spoken answer for the
    # same photo and the same code. a report can't quote a confidence number
    # that a rerun won't reproduce, so the seed is pinned here too.
    torch.manual_seed(0)

    training_set_path = 'data/reference'
    # directories only - macOS silently drops a .DS_Store file into any
    # folder you view in Finder, and a naive os.listdir() would count it
    # as a 4th "class" even though it's not a folder at all. ImageFolder
    # itself only ever looks at real directories, so this has to match
    # that exactly - otherwise class_names.txt ends up off-by-one from
    # what the model actually learned, and every future prediction comes
    # back mislabeled. found this for real: an actual training run built
    # a 4-output classifier from 3 real classes, discovered via ls -la.
    class_names = sorted([d for d in os.listdir(training_set_path)
                           if os.path.isdir(os.path.join(training_set_path, d))])
    n_classes = len(class_names)

    print()
    print('pantry identifier  ·  training')
    print()
    print('  classes   {}'.format(n_classes))
    print('  photos    {}'.format(sum(len(os.listdir(os.path.join(training_set_path, c)))
                                      for c in class_names)))
    model = build_item_classifier(n_classes)
    trainable = sum(p.numel() for p in model.fc.parameters())
    print('  training  {} -> {} ({:,} parameters, everything else frozen)'.format(
        'fc', n_classes, trainable))
    print()

    train_loader = load_item_data(training_set_path)
    # 99% threshold matches greek_transfer.py's real, proven value - stop
    # once training is basically perfect rather than always running the
    # full epoch count
    train_item_classifier(model, [model.fc], train_loader, early_stop_threshold=99.0)

    train_accuracy = evaluate_classifier(model, train_loader)
    print()
    print('  accuracy  {:.0f}% on the photos it trained on'.format(train_accuracy))

    torch.save(model.state_dict(), 'item_classifier.pth')
    with open('class_names.txt', 'w') as f:
        f.write('\n'.join(class_names))
    print('  saved     item_classifier.pth, class_names.txt')
    return

if __name__ == "__main__":
    main(sys.argv)
# main function end
