import sys
from torch.utils.data import DataLoader, SubsetRandomSampler

debug = False

class NOOPMLFlow:
    def set_tracking_uri(self, uri):
        pass
    def set_experiment(self, exp):
        pass
    def start_run(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass

class NOOPMLFlowDataTags:
    def get(self,tag): return "-"

class NOOPMLFlowInfo:
    run_id="0"

class NOOPMLFlowData:
    tags = NOOPMLFlowDataTags()

class NOOPMLFlowRun:
    info = NOOPMLFlowInfo()
    data = NOOPMLFlowData()


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
