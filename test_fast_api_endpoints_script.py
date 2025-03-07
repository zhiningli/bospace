import httpx
import asyncio

async def main():
    url = "http://localhost:8000/boptimise"
    payload = {
        "code_str": """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
X, y = make_classification(
    n_samples=4000,
    n_features=25,
    n_informative=12,
    n_redundant=3,
    n_classes=3,
    class_sep=0.5,
    flip_y=0.1,
    random_state=2
)
numpy_dataset = np.column_stack((X, y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    model = Model(input_size=25, num_classes=3)  # Adjust input size to match your dataset
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
""",
        "search_space": {
            "lower_bound":  {'learning_rate': 0.007196856730011514, 'weight_decay': 0.0, 'num_epochs': 15, 'momentum': 0.05},
            "upper_bound": {'learning_rate': 0.05689866029018293, 'weight_decay': 0.01, 'num_epochs': 85, 'momentum': 0.9500000000000001}
        },
        "n_initial_points": 5,
        "n_iter": 20,
        "allow_logging": True
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            task_id = data["task_id"]
            print(f"✅ Task started successfully! Task ID: {task_id}")
            print("Now listening for updates via WebSocket...")
            return task_id
        else:
            print(f"❌ Error: {response.text}")
            return None

task_id = asyncio.run(main())

# Output task_id so you can use it in JavaScript
print(f"Use this Task ID for WebSocket: {task_id}")