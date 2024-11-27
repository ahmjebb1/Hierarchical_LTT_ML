import sys

from dataset import LTTMLSpectrogramDataset
from models import get_model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import multilabel_confusion_matrix

from util import *
from config import LTTMLConfig

import mlflow
import os

class LTTMLTrainApp:
    def __init__(self, config_file):
        self._config = LTTMLConfig(config_file)

        self._model = get_model(self._config)

        self._config.loss_fn = nn.CrossEntropyLoss()
        self._config.optimizer = optim.Adam(self._model.parameters(), lr=self._config.learning_rate)
        self._config.scheduler = ReduceLROnPlateau(self._config.optimizer, mode='min', factor=0.1, patience=1048576)

        log(0, "Starting ...")
        log(-1, self._model)

        # Set up device
        self._device = torch.device("cuda" if torch.cuda.is_available() and self._config.use_cuda==True else "cpu")

        if torch.backends.mps.is_available():
            log(0, "MPS device available.")
            if self._config.use_mps:
                self._device = torch.device("mps")
            else:
                log(0, "MPS disabled.")
        else:
            log(0, "MPS device not found.")
            # Set device to GPU if available, else use CPU

        log(0, f"Device: {self._device}")
        
        # Move model to the device
        self._model.to(self._device)

        if self._config.use_mlflow == False:
            # Disable automatic logging
            mlflow.start_run = lambda: NOOPMLFlow()  # Mock the start_run function
            mlflow.log_param = lambda run_id, key, value: None  # Mock log_param
            mlflow.log_params = lambda params: None
            mlflow.log_metric = lambda run_id, key=0, value=0, step=0: None  # Mock log_metric
            mlflow.log_artifact = lambda run_id, local_path, artifact_path=None: None  # Mock log_artifact
            mlflow.set_tracking_uri = lambda url: None
            mlflow.set_experiment = lambda name: None
            mlflow.active_run = lambda: NOOPMLFlowRun()
            mlflow.set_tag = lambda tag, desc: None

        mlflow.set_tracking_uri(self._config.mlflow_uri)
        mlflow.set_experiment(self._config.mlflow_expname)

        os.environ["MLFLOW_TRACKING_USERNAME"] = self._config.mlflow_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = self._config.mlflow_pw        

    def load_data(self):

        # Load the whole dataset
        dataset = LTTMLSpectrogramDataset(self._config)
        log(0, f"Loaded data. Total size: {dataset.__len__()} (Applied {self._config.augmentation}x additional augmentation)")

        self._train_loader, self._val_loader, self._test_loader = split_data(dataset, self._config.batch_size)

        # Do this (with no augmentation) to get the stats the first time the dataset is used:
        # https://github.com/YuanGongND/ast/blob/master/src/get_norm_stats.py
        if self._config.calculate_norm_stats:
            log(0, "calculating normalization stats.")
            if self._config.normalize:
                log(1, "calculating normalization stats whilst also normalizing.")
            dataset.get_norm_stats()
            sys.exit(0)

        # Make sure the normalisation in the transformation step is not applied. Can check the
        # numbers are correct by enabling the normalisation in the transformation step and running
        # this function again - it should return a mean of ~0 and std of ~0.5

    def mlflow_logs(self):
        if self._config.use_mlflow == False:
            return
        
        run = mlflow.active_run().info.run_id
        log(0, "mlflow run: "+str(run))
        slurm_id = os.getenv('SLURM_JOB_ID','no_slurm_id')
        log(0, "SLURM id: "+str(slurm_id))
        with open(slurm_id+".mlflow_run", 'w') as f:
            f.write(run)

        params = {
            "epochs": self._config.epochs,
            "learning_rate": self._config.learning_rate,
            "batch_size": self._config.batch_size,
            "min_examples_per_class": self._config.min_examples_per_class,
            "max_examples_per_class": self._config.max_examples_per_class,
            "conv1_size": self._config.conv1_size,
            "conv2_size": self._config.conv2_size,
            "hidden": self._config.hidden,
            "dropout_rate": self._config.dropout_rate,
            "loss_function": self._config.loss_fn.__class__.__name__,
            #"metric_function": metric_fn.__class__.__name__,
            "optimizer": "Adam",
        }
    
        # Description of the MLFlow run:
        mlflow.set_tag("mlflow.note.content", self._config.mlflow_description)

        # update the run name with the prefix:
        run_name = mlflow.active_run().data.tags.get('mlflow.runName')
        new_run_name = self._config.mlflow_runname_prefix+ run_name
        mlflow.set_tag('mlflow.runName', new_run_name)

        # Tags:
        mlflow.set_tag("model_type", self._config.model_type)

        # Can just log the yaml config as params:
        # mlflow.log_params(config)
        
        # And/or save the YAML config file as an artifact:
        mlflow.log_artifact(sys.argv[1], "configs")
                
        # Log training parameters.
        mlflow.log_params(params)

    def forward(self, inputs, labels):

        # create a 1-hot output vector representing the correct label
        one_hot = torch.eye(self._config.num_outputs,dtype=torch.float,device=self._device)[labels]

        # Zero the parameter gradients
        self._config.optimizer.zero_grad()

        # Forward pass
        result = self._model(inputs)
        outputs = result if self._config.model_type != "transformer" else result.logits

        # transform model outputs
        final_outputs = outputs

        if self._config.use_softmax:
            probs = F.softmax(outputs, dim=1)
            final_outputs = probs

        # get loss
        loss = self._config.loss_fn(final_outputs, one_hot)
        return loss, final_outputs
    
    # Validation and Test evaluation:
    def evaluate(self, loader):
        # Set model to evaluation mode
        self._model.eval()

        # Lists to store predictions and ground truth labels
        predictions = []
        true_labels = []

        correct_predictions = 0
        total_samples = 0
        running_loss = 0.0

        # Iterate over the validation set
        for inputs, labels in loader:

            # Forward pass
            # Move inputs and labels to the device
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)
            loss, final_outputs = self.forward(inputs, labels)

            running_loss += loss.item() * inputs.size(0)

            # get predicted labels
            _, preds = torch.max(final_outputs, dim=1)

            # Append predictions and true labels to lists
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

        # Compute stats
        loss = running_loss / total_samples
        accuracy = correct_predictions / total_samples

        return accuracy, loss, true_labels, predictions

    def train(self):
        
        # Training loop
        num_epochs = self._config.epochs

        best_validation_performance = -1.0
        print(f"epoch,train_loss,train_accuracy,val_loss,val_accuracy")

        with mlflow.start_run():
            # Log artifacts and parameters:
            self.mlflow_logs()

            for epoch in range(num_epochs):

                self._model.train()  # Set the model to train mode

                running_loss = 0.0
                correct_predictions = 0
                total_samples = 0
                iteration = 0

                for inputs, labels in self._train_loader:

                    iteration = iteration + 1

                    # Move inputs and labels to the device
                    inputs = inputs.to(self._device)
                    labels = labels.to(self._device)
                    loss, final_outputs = self.forward(inputs, labels)

                    # Backward pass and optimize
                    loss.backward()
                    self._config.optimizer.step()

                    # Compute statistics
                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(final_outputs, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

                # Print statistics for the epoch
                epoch_loss = running_loss / total_samples
                accuracy = correct_predictions / total_samples

                val_accuracy, val_loss, _, _ = self.evaluate(self._val_loader)
                self._config.scheduler.step(val_accuracy)

                lr = self._config.scheduler.get_last_lr()

                mlflow.log_metric("Loss", f"{epoch_loss:4f}", step=(epoch))
                mlflow.log_metric("Accuracy", f"{accuracy:4f}", step=(epoch))
                mlflow.log_metric("Validation accuracy", f"{val_accuracy:4f}", step=(epoch))

                print(f"{epoch},{epoch_loss:.4f},{accuracy:.4f},{val_loss:.4f},{val_accuracy:.4f}")

                if val_accuracy > best_validation_performance:
                    log(0, f"New best validation performance at epoch {epoch+1} of {val_accuracy:.4f} - saving model.")
                    best_validation_performance = val_accuracy
                    torch.save(self._model.state_dict(), self._config.best_model)

            log(0, "Training finished.")

    def test(self):
        log(0, "Beginning final test. Loading best model for evaluation...")
        torch.load(self._config.best_model)
        test_accuracy, test_loss,t_labels, p_labels = self.evaluate(self._test_loader)
        log(0, f"Test-set accuracy: {test_accuracy:.4f} Loss: {test_loss:.4f}")
        log(0, multilabel_confusion_matrix(t_labels, p_labels))
        mlflow.log_metric("Test accuracy", f"{test_accuracy:4f}")
        mlflow.log_metric("Test loss", f"{test_loss:4f}")

def main():
    if len(sys.argv) > 1:
        app = LTTMLTrainApp(sys.argv[1])
        app.load_data()
        app.train()
        app.test()
    else:
        print("Usage: cnn_train.py <config.yaml>")
        exit(0)

if __name__ == "__main__":
    main()

