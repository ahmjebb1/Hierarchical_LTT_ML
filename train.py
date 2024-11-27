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

        dataloader = self._train_loader
        loss_fn = self._config.loss_fn
        optimizer = self._config.optimizer

    def train(self):
        # Training loop
        num_epochs = self._config.epochs

        best_validation_performance = -1.0
        print(f"epoch,train_loss,train_accuracy,val_loss,val_accuracy")

        os.environ["MLFLOW_TRACKING_USERNAME"] = self._config.mlflow_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = self._config.mlflow_pw
        mlflow.set_tracking_uri(self._config.mlflow_uri)
        mlflow.set_experiment(self._config.mlflow_expname)
    
        with mlflow.start_run():
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

            for epoch in range(num_epochs):

                self._model.train()  # Set the model to train mode

                running_loss = 0.0
                correct_predictions = 0
                total_samples = 0
                iteration = 0

                for inputs, labels in self._train_loader:

                    # shape = [batch, channels, length, bins], e.g. [3, 1, 300, 128]
                    # AST expects [batch, channels, length, bins] but ends up with input of size: [3, 1, 300, 1, 128]
                    # in AST old code [4, 1024, 128]! => must be adding channels in by itself somehow
                    # if _model_type=="transformer": # removing channels seems to work:
                    #     inputs = inputs.squeeze(1)

                    iteration = iteration + 1
                    # Move inputs and labels to the device

                    inputs = inputs.to(self._device)
                    labels = labels.to(self._device)

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

                val_accuracy, val_loss, _, _ = validate(self._config, self._model, self._val_loader, self._device, self._config.loss_fn)
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
        log(0, "Beginning final test.\nLoading best model...")
        torch.load(self._config.best_model)
        test_accuracy, test_loss,t_labels, p_labels = validate(self._config, self._model, self._test_loader, self._device, self._config.loss_fn)
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

