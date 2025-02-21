""" This service is for evaluating a model or a dataset's response to a set of predefined hyperparameter configurations. 
This information is crucial for evaluating the rank correlation between two datasets or two models.

Results for saved to the HPO-evaluation table.
"""

from src.database.crud import ModelRepository
from src.database.crud import DatasetRepository
from src.database.crud import HPEvaluationRepository
import numpy as np
import torch

class HPEvalutaionService:

    def __init__(self):

        self.model_candidates_index = []
        self.dataset_candidates_index = []
        self.manual_seed = 42
        self.search_space = {
            'learning_rate': np.logspace(-5, -1, num=50).tolist(),  # Logarithmically spaced values
            'momentum': [0.01 * x for x in range(100)],  # Linear space
            'weight_decay': [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'num_epochs': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90],}
        self.bounds = torch.Tensor([
                            [0, 0, 0, 0],
                            [len(self.search_space['learning_rate'])-1, 
                            len(self.search_space['momentum'])-1, 
                            len(self.search_space['weight_decay'])-1, 
                            len(self.search_space['num_epochs'])-1]], )

    def run_hp_evaluations_for_all_models():

        for model_idx in range(102, 104):
            model_module = importlib.import_module(f"src.scripts.models.model{model_idx}")
            model = getattr(model_module, "model", None)
            for dataset_idx in range(6, 17):
                dataset_module = importlib.import_module(f"src.scripts.datasets.dataset{dataset_idx}")
                dataset = getattr(dataset_module, "dataset", None)
                input_size = getattr(dataset_module, "input_size")
                num_classes = getattr(dataset_module, "num_classes")
                code_str = f"""
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.datasets import make_classification
        {dataset}
        {model}
        def train_simple_nn(learning_rate, momentum, weight_decay, num_epochs):
            # Initialize model, loss function, and optimizer
            model = Model(input_size={input_size}, num_classes={num_classes})  # Adjust input size to match your dataset
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
                print(f"Model idx {model_idx} Dataset idx {dataset_idx}")
                store.code_string = code_str
                store.instantiate_code_classes()
                print(store.dataset_instance.shape)
                labels = store.dataset_instance[:, -1]  # Extract last column (labels)
                unique_labels = np.unique(labels)  # Find unique labels
                num_labels = len(unique_labels) 
                print("Labels", num_labels)

                train_y = []
                for x in tqdm(train_x, desc="Inital sampling"):
                    train_y.append(botorch_objective(x, store).item())

                script_object = {
                    "script": code_str,
                    "model_idx": model_idx,
                    "dataset_idx": dataset_idx,
                    "script_idx": dataset_idx * 1000 + model_idx,
                    "results": {
                        str(idx): {
                            "learning_rate": search_space["learning_rate"][int(train_x[idx][0].item())],
                            "momentum": search_space["momentum"][int(train_x[idx][1].item())],
                            "weight_decay": search_space["weight_decay"][int(train_x[idx][2].item())],
                            "num_epochs": search_space["num_epochs"][int(train_x[idx][3].item())],
                            "accuracy": train_y[idx]  # Store accuracy as the evaluation metric
                        }
                        for idx in range(len(train_y))
                    }
                }

                script_repo.save_scripts(script_object)

