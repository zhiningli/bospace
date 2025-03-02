code_str = """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import openml

    
dataset = openml.datasets.get_dataset(dataset_id=554)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
target_names = dataset.retrieve_class_labels() if dataset.retrieve_class_labels() else None

# Limit dataset to 5,000 samples if larger
max_samples = 5000
if len(X) > max_samples:
    X, _, y, _ = train_test_split(X, y, train_size=max_samples, stratify=y, random_state=42)

# Encode categorical columns
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# Step 4: Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X=X)
y = y.to_numpy()
# === Encode Labels if Needed ===
if y.dtype.kind not in "iu":  # Convert non-integer labels to numeric
    y = LabelEncoder().fit_transform(y)
numpy_dataset = np.column_stack((X, y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# **Step 2: Convert to PyTorch tensors**
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# **Step 3: Create PyTorch Datasets**
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# **Step 4: Create DataLoaders**
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) 


class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * (input_size // 2), 128)  # Adjust for the reduced dimension after pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Add regularization with dropout

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def train_simple_nn(learning_rate, momentum, weight_decay, num_epochs):
    # Initialize model, loss function, and optimizer
    model = Model(input_size=784, num_classes=10)  # Adjust input size to match your dataset
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
    # Testing the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy
"""


from src.middleware import ComponentStore
import numpy as np

store = ComponentStore()
store.code_string = code_str
store.instantiate_code_classes()

train_y = []
support = []
learning_rate_range = np.logspace(-5, -0.5, num=25).tolist()

num_epoch_range = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100]
for learning_rate in learning_rate_range:
    for num_epoch in num_epoch_range:
        print(f"Learning rate: {learning_rate}, Num Epochs: {num_epoch}")
        kwargs = {
            "learning_rate": learning_rate,
            "momentum": 0.9,
            "weight_decay": 0.001,
            "num_epochs": num_epoch,
        }

        avg_eval_accuracy = 0
        for i in range(3):

            eval_accuracy = store.objective_func(**kwargs)
            avg_eval_accuracy += eval_accuracy
            print(f"Iteration {i}, Accuracy: {avg_eval_accuracy}")
        avg_eval_accuracy/=3         
        print(f"Average Accuracy: {avg_eval_accuracy}\n")
        
       
        support.append((learning_rate, num_epoch, avg_eval_accuracy))

import csv

with open("example_result.csv", "w", newline="") as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(["Learning Rate", "Num Epochs", "Accuracy"])  # Write header
    
    for entry in support:
        csv_writer.writerow(entry)  # Write data rows




