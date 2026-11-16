# Spoken Pantry Identifier
# final project: accessible pantry item identifier
# VIEW layer - visual verification plots for ME, the developer
# Does: plots detected items with an index, plots a reference session
#       labeled by assigned item name, plots per-item accuracy for the
#       final evaluation
# Does not: get seen by the end user at all - that's speech_view.py's job
# Reasoning: still a view in the mvc sense - takes data the model produced
#       and decides how to present it, just to a different audience (a
#       screen, not an ear) and for a different purpose (catching a bad
#       detection before trusting it, not delivering the accessibility
#       feature itself). same "check before you trust it" plots already
#       used for digits and greek letters in project 5.

# imports start
import cv2
import matplotlib.pyplot as plt
# imports end


# useful functions start

# plots every detected item with its index, to visually confirm detection
# and order before trusting it - used when segmenting a single query photo,
# where there's no class label yet (that's the whole question being asked)
def plot_detected_items(crops, save_path='detected_items_check.png'):
    n = len(crops)
    if n == 0:
        print('*** nothing to plot - zero items were detected')
        return

    plt.figure(figsize=(min(15, 2.2 * n), 3))
    for i, crop in enumerate(crops):
        plt.subplot(1, n, i + 1)
        plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        plt.title(str(i))
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# plots every detected crop labeled with the item name it was assigned, so
# a wrong layout/miscount is visible before trusting the saved reference
# files - used during reference-capture sessions, where CLASS_ORDER gives
# every crop a name
def plot_labeled_session(crops, class_order, save_path='reference_session_check.png'):
    n = len(crops)
    if n == 0:
        print('*** nothing to plot - zero items were detected')
        return

    plt.figure(figsize=(min(18, 2.2 * n), 3))
    for i, crop in enumerate(crops):
        plt.subplot(1, n, i + 1)
        plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        label = class_order[i] if i < len(class_order) else '?'
        plt.title(label, fontsize=8, rotation=30, ha='right')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# plots per-item accuracy for a batch of held-out query photos.
# all_classes is the full class list, and every class in it gets a slot even
# when it has no query photograph. that matters: the earlier version plotted
# only the classes it had data for, so six of the nine simply vanished from the
# chart, and a reader had no way to tell a missing class from a class that was
# never tested. an absent bar reads as "not applicable"; a labelled empty slot
# reads as "no measurement exists", which is the true statement.
def plot_confusion_summary(predictions, actuals, all_classes=None,
                           save_path='prediction_summary.png'):
    labels = list(all_classes) if all_classes else sorted(set(actuals))

    correct_counts = {label: 0 for label in labels}
    total_counts = {label: 0 for label in labels}
    for pred, actual in zip(predictions, actuals):
        if actual not in total_counts:
            continue
        total_counts[actual] += 1
        if pred == actual:
            correct_counts[actual] += 1

    tested = [l for l in labels if total_counts[l] > 0]
    untested = [l for l in labels if total_counts[l] == 0]

    plt.figure(figsize=(max(9, len(labels) * 1.05), 4.2))
    for i, l in enumerate(labels):
        if total_counts[l] > 0:
            plt.bar(i, 100. * correct_counts[l] / total_counts[l],
                    color='#0e7c86', width=0.62)
            plt.text(i, 100. * correct_counts[l] / total_counts[l] + 3,
                     '{}/{}'.format(correct_counts[l], total_counts[l]),
                     ha='center', fontsize=8, color='#0a5a62')
        else:
            # an empty slot, outlined and labelled - not a bar of height zero
            plt.bar(i, 100, color='none', edgecolor='#c9483f', width=0.62,
                    linestyle='--', linewidth=1.2)
            plt.text(i, 50, 'no query\nphotograph', ha='center', va='center',
                     fontsize=7, color='#c9483f')

    plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=8)
    plt.ylabel('accuracy (%)')
    plt.ylim(0, 112)
    plt.title('per-item top-1 accuracy   ·   {} of {} classes measured'.format(
        len(tested), len(labels)), fontsize=10, loc='left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
# useful functions end
