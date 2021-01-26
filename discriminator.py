import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from symmetric import Symmetric


class Discriminator(nn.Module):
    def __init__(self):
        """Build the discriminator architecture"""
        super(Discriminator, self).__init__()

        # 1 because it is only 1 channel in a tensor (N, C, H, W)
        self.batch1 = nn.BatchNorm2d(1, eps=0.001, momentum=0.99)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(1, 7),
            stride=(1, 2),
            padding=(0, 3),
        )
        self.batch2 = nn.BatchNorm2d(64, eps=0.001, momentum=0.99)

        self.symm1 = Symmetric("mean", 2)

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(1, 7),
            stride=(1, 2),
            padding=(0, 3),
        )
        self.batch3 = nn.BatchNorm2d(128, eps=0.001, momentum=0.99)
        self.dropout1 = nn.Dropout2d(0.5)

        self.symm2 = Symmetric("mean", 3)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    # x represents our data
    def forward(self, x):
        """Mark the flow of data throughout the network"""

        x = self.batch1(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch2(x)

        x = self.symm1(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.batch3(x)
        x = self.dropout1(x)

        x = self.symm2(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = torch.sigmoid(x)

        return output

    def weights_init(self, m):
        """Reset parameters and initialize with random weight values"""

        if isinstance(m, nn.Conv2d):
            m.reset_parameters()
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        if isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
            m.reset_parameters()

    def get_accuracy(self, y_true, y_prob):
        """Compute model accuracy over labelled data"""

        y_true = y_true.squeeze()
        y_prob = y_prob.squeeze()
        y_prob = y_prob > 0.5
        return (y_true == y_prob).sum().item() / y_true.size(0)

    def fit(self, trainflow, valflow, epochs, lr):
        """Train the discriminator model with the Binary Cross-Entropy loss.
        trainflow: PyTorch data loader for the training dataset
        valflow: PyTorch data loader for the validation dataset
        epochs: Number of iterations through the training dataset
        lr: Learning rate for gradient descent with Adam
        """

        optimizer = torch.optim.Adam(self.parameters(), lr)
        lossf = nn.BCELoss()
        best_val_loss = 1.0

        print("Initializing weights of the model, deleting previous ones")
        self.apply(self.weights_init)
        self.train()

        # Loop over the dataset multiple times
        for epoch in range(epochs):

            train_loss, val_loss, acc_train, acc_val = 0.0, 0.0, 0.0, 0.0

            # For each batch of training data
            for i, (inputs, labels) in enumerate(trainflow, 1):

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Compute model predictions, compute loss and perform back-prop
                out = self(inputs)
                loss = lossf(out, labels)
                loss.mean().backward()
                optimizer.step()

                # Print statistics
                train_loss += loss.item()
                acc_train += self.get_accuracy(labels, out)
                if i % 20 == 0:  # print every 20 mini-batches
                    print(
                        "[%d | %d] TRAINING: loss: %.3f | acc: %.3f"
                        % (
                            epoch + 1,
                            i,
                            train_loss / i,
                            acc_train / i,
                        ),
                        end="\r",
                    )

            print("")
            # Calculate stats on validation data with no gradient descent
            with torch.no_grad():
                # For each batch of validation data
                for j, (genmats, labels) in enumerate(valflow, 1):
                    # Compute model predictions, compute loss and stats
                    preds = self(genmats)
                    val_loss += lossf(preds, labels).item()
                    acc_val += self.get_accuracy(labels, preds)
                    print(
                        "        VALIDATION: loss: %.3f - acc: %.3f"
                        % (
                            val_loss / j,
                            acc_val / j,
                        ),
                        end="\r",
                    )
                # Save the model weights with the lowest validation error
                if (val_loss / j) < best_val_loss:
                    best_val_loss = val_loss / j
                    best_train_acc = acc_train / len(trainflow)
                    best_epoch = epoch + 1
                    best_model = copy.deepcopy(self.state_dict())

            print("")

        # Load the model with the lowest validation error
        self.load_state_dict(best_model)
        print(f"Best model has validation loss {best_val_loss:.3f} from {best_epoch}")
        return best_train_acc

    def predict(self, inputs):
        """Compute model prediction over inputs"""
        self.eval()
        with torch.no_grad():
            preds = self(inputs)
        return preds
