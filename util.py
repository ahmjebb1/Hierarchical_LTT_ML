import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler

debug = False

def log(level, *args, **kwargs):
    if level==-1 and not debug: return
    prefix = "??? ("+str(level)+"):\t"
    if level==-1:prefix = "Debug:\t"
    if level==0: prefix = "Info:\t"
    if level==1: prefix = "Warn:\t"
    if level==2: prefix = "Error:\t"
    eprint(prefix, end='')
    eprint(*args, **kwargs)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def split_data(dataset, batch_size):
    # Get the lists of train, validation and test indices within the dataset:
    train_indices = dataset.get_training_indices()
    val_indices = dataset.get_validation_indices()
    test_indices = dataset.get_test_indices()

    log(0, f"Training examples: {train_indices.__len__()} Validation set: {val_indices.__len__()} Test set: {test_indices.__len__()}")

    # Create DataLoader instances for the training, validation, and test sets
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices))
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices))

    return (train_loader, val_loader, test_loader)


def validate(config, model, loader, device, loss_fn):

    # Set model to evaluation mode
    model.eval()

    # Lists to store predictions and ground truth labels
    predictions = []
    true_labels = []

    correct_predictions = 0
    total_samples = 0
    running_loss = 0.0

    # Iterate over the validation set
    for inputs, labels in loader:

        # Forward pass
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():  # No need to track gradients during validation
            result = model(inputs)

        one_hot = torch.eye(config.get('num_outputs'),dtype=torch.float,device=device)[labels]

        result = model(inputs)
        outputs = result if config.get("model") != "transformer" else result.logits

        # transform model outputs
        final_outputs = outputs

        if config.get('use_softmax'):
            probs = F.softmax(outputs, dim=1)
            final_outputs = probs

        # get loss
        loss = loss_fn(final_outputs, one_hot)

        running_loss += loss.item() * inputs.size(0)

        # get predicted labels
        _, preds = torch.max(final_outputs, dim=1)

        # Append predictions and true labels to lists
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)

    # Compute stats
    val_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples

    return accuracy, val_loss, true_labels, predictions
