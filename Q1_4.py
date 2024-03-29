from torchvision import datasets, transforms
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import LeNet
from torchinfo import summary
from torchmetrics import Accuracy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
import os

torch.manual_seed(42)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")  # For M1 chip, torch.device("mps") is used instead of torch.device("cuda") for gpu training.

MODEL_PATH = Path("./models/")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = MODEL_PATH / "lenet_mnist.pth"

class Args:
    batch_size = 32
    epochs = 10  # Set a lower number for quick testing
    print_model = "False"
    mode = 'train'  # Adding the 'mode' attribute with default value 'train'
    learning_rate = 0.001  # Default value; will be overridden by parameter_combinations
    weight_decay = 0.0001  # Default value; will be overridden by parameter_combinations
    hidden_linear = 120  # Default value; will be overridden by parameter_combinations
    hidden_channel = 16  # Default value; will be overridden by parameter_combinations

args = Args()
def preprocess(BATCH_SIZE=32, DEVICE=DEVICE):
    # Download the MNIST dataset

    normalize = transforms.Normalize((0.1307,), (0.3081,))
    mnist_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    train_val_dataset = datasets.MNIST(root="./datasets/", train=True, download=True, transform=mnist_transforms)
    test_dataset = datasets.MNIST(root="./datasets/", train=False, download=True, transform=mnist_transforms)

    # Split the dataset into train and validation
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset=train_val_dataset,
                                                               lengths=[train_size, val_size])

    # Create dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader


def train(model, train_dataloader, val_dataloader, num_epochs, learning_rate, weight_decay, MODEL_SAVE_PATH, DEVICE):
    # Loss function, optimizer and accuracy
    model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    accuracy = Accuracy(task='multiclass', num_classes=10).to(DEVICE)

    writer = SummaryWriter()
    for epoch in tqdm(range(num_epochs)):
        # Training loop
        train_loss, train_acc = 0.0, 0.0
        for X, y in train_dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            model.train()

            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            acc = accuracy(y_pred, y)
            train_acc += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        # Validation loop
        val_loss, val_acc = 0.0, 0.0
        model.eval()
        with torch.inference_mode():
            for X, y in val_dataloader:
                X, y = X.to(DEVICE), y.to(DEVICE)

                y_pred = model(X)

                loss = loss_fn(y_pred, y)
                val_loss += loss.item()

                acc = accuracy(y_pred, y)
                val_acc += acc

            val_loss /= len(val_dataloader)
            val_acc /= len(val_dataloader)
        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train/loss": train_loss, "val/loss": val_loss},
                           global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train/acc": train_acc, "val/acc": val_acc},
                           global_step=epoch)
        print(
            f"Epoch: {epoch}| Train loss: {train_loss: .5f}| Train acc: {train_acc: .5f}| Val loss: {val_loss: .5f}| Val acc: {val_acc: .5f}")

    # Saving the model
    print(f"Saving the model: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)


def test(model, test_dataloader, DEVICE):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()  # Define the loss function; assuming CrossEntropyLoss for classification
    with torch.no_grad():  # No need to track gradients for testing
        for X, y in test_dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y).item()  # Sum up batch loss
            pred = y_pred.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()
            total += y.size(0)

    test_loss /= len(test_dataloader.dataset)
    test_accuracy = 100. * correct / total
    return test_loss, test_accuracy


def evaluate(model, val_dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


if __name__ == "__main__":
    # Define the combinations of parameters to try
    parameter_combinations = [
        {'learning_rate': 0.001, 'weight_decay': 0.0001, 'hidden_linear': 120, 'hidden_channel': 16},
        {'learning_rate': 0.001, 'weight_decay': 0.0001, 'hidden_linear': 100, 'hidden_channel': 12},
        {'learning_rate': 0.001, 'weight_decay': 0.00001, 'hidden_linear': 120, 'hidden_channel': 16},
        {'learning_rate': 0.001, 'weight_decay': 0.00001, 'hidden_linear': 100, 'hidden_channel': 12},
        {'learning_rate': 0.0001, 'weight_decay': 0.0001, 'hidden_linear': 120, 'hidden_channel': 16},
        {'learning_rate': 0.0001, 'weight_decay': 0.0001, 'hidden_linear': 100, 'hidden_channel': 12},
        {'learning_rate': 0.0001, 'weight_decay': 0.00001, 'hidden_linear': 120, 'hidden_channel': 16},
        {'learning_rate': 0.0001, 'weight_decay': 0.00001, 'hidden_linear': 100, 'hidden_channel': 12},
    ]

    # Placeholder for storing the best validation accuracy and corresponding parameters
    best_val_acc = 0
    best_params = None
    best_model_summary = ""

    for params in parameter_combinations:
        # Update model and training parameters
        args.learning_rate = params['learning_rate']
        args.weight_decay = params['weight_decay']
        args.hidden_linear = params['hidden_linear']
        args.hidden_channel = params['hidden_channel']

        # Initialize the model with the current set of parameters
        modellenet = LeNet(hidden_channel=args.hidden_channel, hidden_linear=args.hidden_linear)

        # Download the MNIST dataset and create dataloaders
        train_dataloader, val_dataloader, test_dataloader = preprocess(args.batch_size)

        # Check if the mode is set to train to avoid retraining during testing
        if args.mode == 'train':
            # Train the model
            train(modellenet, train_dataloader, val_dataloader, args.epochs, args.learning_rate, args.weight_decay,
                  MODEL_SAVE_PATH, DEVICE)

            # Evaluate the model on the validation set and update the best model if applicable
            current_val_acc = evaluate(modellenet, val_dataloader,
                                       DEVICE)  # Assuming there's an evaluate function that returns validation accuracy
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                best_params = params
                # Capture the model summary for the best model so far
                best_model_summary = summary(modellenet, input_size=(1, 1, 28, 28), verbose=0, col_width=20,
                                             col_names=["input_size", "output_size", "num_params", "trainable"])

    print(f"Best validation accuracy: {best_val_acc}")
    print(f"Best parameters: {best_params}")
    print("Best model summary:")
    print(best_model_summary)
    # After identifying the best parameters...
    modellenet_best = LeNet(hidden_channel=best_params['hidden_channel'], hidden_linear=best_params['hidden_linear'])
    modellenet_best.load_state_dict(torch.load(MODEL_SAVE_PATH))
    modellenet_best.to(DEVICE)

    # Prepare the test dataset
    _, _, test_dataloader = preprocess(args.batch_size, DEVICE)

    # Test the model
    test_loss, test_accuracy = test(modellenet_best, test_dataloader, DEVICE)

    # Print or log the test results
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")