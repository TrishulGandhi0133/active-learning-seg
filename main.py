import os
from data.prepare_data import prepare_cifar100_data
from models.yolov5_seg import train_yolov5
from active_learning.margin_sampling import margin_sampling
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

def evaluate_model(model, val_data, batch_size=16):
    model.eval()
    dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    total, correct = 0, 0
    for images, labels in dataloader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def active_learning_pipeline(root_dir='data/cifar100', epochs=10, batch_size=16, n_queries=10):
    # Step 1: Prepare data
    print("Preparing CIFAR-100 data...")
    train_data, test_data = prepare_cifar100_data(root_dir)
    print("Data preparation completed.\n")

    # Step 2: Initial training
    print("Starting initial training...")
    model = train_yolov5(train_data, test_data, epochs, batch_size)
    initial_accuracy = evaluate_model(model, test_data, batch_size)
    print(f"Initial training completed. Validation accuracy: {initial_accuracy:.2f}%\n")

    # Step 3: Active Learning Loop
    for cycle in range(5):
        print(f"Active Learning Cycle {cycle + 1}...")
        # Get model predictions
        dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        predictions = []
        for images, _ in dataloader:
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            predictions.extend(probabilities)
        predictions = np.array(predictions)
        
        # Select samples based on margin sampling
        print("Selecting samples using margin sampling...")
        query_indices = margin_sampling(predictions, n_queries)
        print(f"Selected {n_queries} samples.\n")

        # Add queried samples to training data
        queried_data = Subset(test_data, query_indices)
        train_data = torch.utils.data.ConcatDataset([train_data, queried_data])
        
        # Retrain the model
        print("Retraining the model...")
        model = train_yolov5(train_data, test_data, epochs, batch_size)
        accuracy = evaluate_model(model, test_data, batch_size)
        print(f"Cycle {cycle + 1} completed. Validation accuracy: {accuracy:.2f}%\n")

if __name__ == "__main__":
    active_learning_pipeline()
