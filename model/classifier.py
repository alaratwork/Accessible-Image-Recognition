# Spoken Pantry Identifier
# final project: accessible pantry item identifier
# MODEL layer - the classifier itself: network definition, training loop,
# and data loading
# Does: builds a frozen+swapped resnet18, or a configurable variant with
#       different unfrozen depth; loads reference data with the right
#       transform; trains and evaluates a model given any list of
#       trainable modules
# Does not: decide WHEN to train, print anything about training progress
#       on its own initiative, or know who's calling it - training_
#       controller and experiment_controller both call into this file,
#       for different reasons
# Reasoning: direct port of greek_transfer.py's freeze-and-swap pattern
#       (cs5330 project 5): freeze every existing parameter, then train
#       only a newly attached piece. two things changed from that
#       original version, each for a specific reason:
#       - the backbone is a pretrained resnet18, not the mnist-trained
#         MyNetwork - pantry photos are natural color images with real
#         texture and lighting, mnist's stroke-based features have
#         nothing useful to transfer from
#       - train_item_classifier takes a LIST of trainable modules instead
#         of hardcoding fc2 - lets one function serve both the production
#         path (fc only) and the experiment controller's fine-tuning-depth
#         comparison (fc_and_layer4, fc_layer4_layer3), instead of
#         duplicating the training loop for each

# imports start
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # must be set before importing torch - same macOS fix project 5 needed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
# imports end


# useful functions start

# loads a pretrained resnet18, freezes every existing parameter, and swaps
# the final layer for a fresh one sized to the actual item count - same
# freeze-then-replace order as greek_transfer.py's build_greek_network()
def build_item_classifier(n_classes):
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    # resnet18's final layer is called fc, not fc2 - same idea as MyNetwork,
    # different attribute name
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


# same as build_item_classifier, but with a configurable number of unfrozen
# layers - fc only, or also the last one or two resnet blocks - used by the
# experiment controller to compare fine-tuning depth. kept in this file
# rather than the controller because "what does the network look like" is
# a model concern regardless of who's asking for a variant of it.
def build_configurable_classifier(n_classes, unfrozen='fc_only'):
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    trainable_modules = [model.fc]
    if unfrozen in ('fc_and_layer4', 'fc_layer4_layer3'):
        for param in model.layer4.parameters():
            param.requires_grad = True
        trainable_modules.append(model.layer4)
    if unfrozen == 'fc_layer4_layer3':
        for param in model.layer3.parameters():
            param.requires_grad = True
        trainable_modules.append(model.layer3)

    return model, trainable_modules


# standard imagenet preprocessing - matches what the pretrained backbone
# was itself trained on
def build_transform(image_size=224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# builds a dataloader for the item training folder - one subfolder per
# item, same ImageFolder pattern as greek_transfer.py's load_greek_data()
def load_item_data(training_set_path, batch_size=8):
    return torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(training_set_path, transform=build_transform()),
        batch_size=batch_size,
        shuffle=True
    )


# splits the reference folder into a train/validation split - holds out 1
# photo per item for validation, trains on the rest. used by the experiment
# controller so hyperparameter tuning never touches the 20 separately-shot
# query photos, which stay reserved for the one final reported evaluation.
def build_train_val_loaders(reference_dir, batch_size, val_holdout_per_class=1):
    full_dataset = torchvision.datasets.ImageFolder(reference_dir, transform=build_transform())

    val_indices, train_indices = [], []
    seen_per_class = {}
    for idx, (_, label) in enumerate(full_dataset.samples):
        seen_per_class[label] = seen_per_class.get(label, 0)
        if seen_per_class[label] < val_holdout_per_class:
            val_indices.append(idx)
        else:
            train_indices.append(idx)
        seen_per_class[label] += 1

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=8, shuffle=False)
    return train_loader, val_loader, len(full_dataset.classes)


# trains only the unfrozen parameters of a model for n_epochs - same
# pattern as greek_transfer.py's train_greek_network(), generalized to
# accept any list of trainable modules instead of always just fc2.
# early_stop_threshold reproduces greek_transfer.py's real behavior
# (stopped fine-tuning once training accuracy hit 99%, logged the epoch
# count for the report) - left as opt-in, not a hardcoded default, because
# the experiment controller specifically needs every configuration trained
# for the SAME number of epochs to compare them fairly. the training
# controller passes a threshold; the experiment controller doesn't.
def train_item_classifier(model, trainable_modules, train_loader, n_epochs=15,
                           learning_rate=0.01, momentum=0.5, early_stop_threshold=None):
    params = []
    for module in trainable_modules:
        params += list(module.parameters())
    optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum)

    losses = []
    model.train()
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        correct = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100. * correct / len(train_loader.dataset)
        losses.append(avg_loss)
        print('  epoch {:<2}  loss {:.3f}   accuracy {:>3.0f}%'.format(epoch, avg_loss, accuracy))

        if early_stop_threshold is not None and accuracy >= early_stop_threshold:
            print('  stopped early - reached {:.0f}% after {} epochs'.format(early_stop_threshold, epoch))
            break

    return losses


# evaluates a model on a dataloader, returns accuracy - shared by both the
# training controller (sanity check after training) and the experiment
# controller (scoring each swept configuration)
def evaluate_classifier(model, data_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
    return 100. * correct / len(data_loader.dataset)
# useful functions end
