import gc

import torch
import torch.amp as amp
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# the network is optimized using the Adam
# optimizer applying binary cross entropy (19) and learning rate
# l = 0.001
# It also deploys a batch size of 128 instances
# to train the complete network. Every NN model has trained
# about 100 epochs within the experiments, followed by the
# early stopping point for reporting the results.


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, lr=0.001, num_epochs=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.lr = lr

        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.best_val_loss = float("inf")
        self.best_model_state = None

    def clear_memory(self):
        """Clear CUDA cache and run garbage collector"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

        for batch_idx, (main_texts, texts, images, labels) in enumerate(pbar):
            main_texts, texts, images, labels = (
                main_texts.to(self.device, non_blocking=True),
                texts.to(self.device, non_blocking=True),
                images.squeeze(1).to(self.device, non_blocking=True),  # Remove unnecessary dimension
                labels.to(self.device, non_blocking=True),
            )

            self.optimizer.zero_grad()
            outputs = self.model(main_texts, texts, images)
            loss = self.criterion(outputs, labels)

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Prevent exploding gradients
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if batch_idx % 10 == 0:  # Free memory every 10 batches
                self.clear_memory()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        for main_texts, texts, images, labels in self.val_loader:
            main_texts, texts, images, labels = (
                main_texts.to(self.device, non_blocking=True),
                texts.to(self.device, non_blocking=True),
                images.squeeze(1).to(self.device, non_blocking=True),
                labels.to(self.device, non_blocking=True),
            )

            outputs = self.model(main_texts, texts, images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds)
            all_labels.append(labels)

        # Concatenate predictions & labels to avoid excessive `.cpu().numpy()` calls
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()

        avg_loss = total_loss / len(self.val_loader.dataset)
        metrics = self.calculate_metrics(all_preds, all_labels)

        self.clear_memory()
        return avg_loss, metrics

    @staticmethod
    def calculate_metrics(predictions, labels):
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def print_metrics(metrics, phase):
        print(f"\n{phase} Metrics:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        print("-" * 50)

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate()

            print(f"\nEpoch {epoch + 1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")
            self.print_metrics(val_metrics, "Validation")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

    def test(self):
        if self.best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
        test_loss, test_metrics = self.validate()
        print("\nBest Model Performance on Test Set:")
        self.print_metrics(test_metrics, "Test")
