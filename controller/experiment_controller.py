# Spoken Pantry Identifier
# final project: accessible pantry item identifier
# CONTROLLER layer - round-robin coordinate-ascent hyperparameter search
# Does: tries one hyperparameter value at a time, scores it, keeps the
#       best value for that dimension, moves to the next dimension,
#       repeats for 3 full rounds, saves every result to csv
# Does not: define the network itself - it builds every model it tests
#       via model/classifier.py, never redefining the architecture here
# Reasoning: same proven pattern as cs5330 project 5's experiment.py, but
#       the 3 swept dimensions are different on purpose - a frozen
#       pretrained resnet18 has no "number of filters" to tune, so this
#       sweeps learning rate, how many layers stay unfrozen, and batch
#       size instead - the parameters that actually matter here.

# imports start
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # must be set before importing torch
import sys
import csv
import torch
from model.classifier import build_configurable_classifier, build_train_val_loaders, train_item_classifier, evaluate_classifier
# imports end


# useful functions start

# trains and evaluates one configuration, returns validation accuracy.
# deliberately does not pass early_stop_threshold - every configuration
# must train for the same n_epochs for the comparison across configs to
# mean anything, same reasoning the original experiment.py's design used
def run_one_config(learning_rate, unfrozen, batch_size, reference_dir, n_epochs=5):
    torch.manual_seed(0)
    train_loader, val_loader, n_classes = build_train_val_loaders(reference_dir, batch_size)
    model, trainable_modules = build_configurable_classifier(n_classes, unfrozen)
    train_item_classifier(model, trainable_modules, train_loader, n_epochs=n_epochs, learning_rate=learning_rate)
    return evaluate_classifier(model, val_loader)


# writes the full sweep results to a csv file for the report
def save_results(results, path='experiment_results.csv'):
    fieldnames = ['round', 'dimension', 'value', 'accuracy', 'learning_rate', 'unfrozen', 'batch_size']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
# useful functions end


# main function start

# sweep configuration - 3 dimensions x rounds, same round-robin
# coordinate-ascent pattern as p5, carrying each dimension's best value
# forward into the next sweep
LR_OPTIONS = [0.001, 0.01, 0.05, 0.1]
UNFROZEN_OPTIONS = ['fc_only', 'fc_and_layer4', 'fc_layer4_layer3']
BATCH_OPTIONS = [4, 8, 16]

BASELINE = {'learning_rate': 0.01, 'unfrozen': 'fc_only', 'batch_size': 8}
DIMENSION_ORDER = ['learning_rate', 'unfrozen', 'batch_size']
DIMENSION_OPTIONS = {
    'learning_rate': LR_OPTIONS,
    'unfrozen': UNFROZEN_OPTIONS,
    'batch_size': BATCH_OPTIONS,
}


# runs the round-robin search: sweeps one dimension at a time, holding the
# other two at the current best-known values, then carries the winner
# forward - same logic already proven (with a stand-in accuracy function)
# during the mvc restructure
def run_experiment(reference_dir, n_rounds=3):
    current = dict(BASELINE)
    results = []
    convergence = []

    for round_num in range(1, n_rounds + 1):
        for dim in DIMENSION_ORDER:
            best_value = current[dim]
            best_acc = -1.0

            for value in DIMENSION_OPTIONS[dim]:
                trial = dict(current)
                trial[dim] = value
                acc = run_one_config(trial['learning_rate'], trial['unfrozen'],
                                      trial['batch_size'], reference_dir)
                results.append({
                    'round': round_num, 'dimension': dim, 'value': value, 'accuracy': acc,
                    'learning_rate': trial['learning_rate'], 'unfrozen': trial['unfrozen'],
                    'batch_size': trial['batch_size'],
                })
                if acc > best_acc:
                    best_acc = acc
                    best_value = value

            current[dim] = best_value
            convergence.append({'round': round_num, 'dimension': dim,
                                 'best_value': best_value, 'best_accuracy': best_acc})
            print('round {} - best {}: {} (accuracy {:.1f}%)'.format(round_num, dim, best_value, best_acc))

    print('final configuration after {} rounds: {}'.format(n_rounds, current))
    return results, convergence


# sweeps 3 dimensions x 3+ values x 3 rounds against the reference set's
# train/val split - the 20 separately-shot query photos are never touched
# here, reserved for the final reported evaluation
def main(argv):
    reference_dir = 'data/reference'
    results, convergence = run_experiment(reference_dir)
    save_results(results)
    return

if __name__ == "__main__":
    main(sys.argv)
# main function end
