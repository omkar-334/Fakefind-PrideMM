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
        total_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

        for main_texts, texts, images, labels in pbar:
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache())
            main_texts = main_texts.to(self.device, non_blocking=True)
            texts = texts.to(self.device, non_blocking=True)
            images = images.to(self.device, non_blocking=True)
            # print(images.shape)
            images = images.squeeze(1)  # Removes the dimension with size 1
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(main_texts, texts, images)
            loss = self.criterion(outputs, labels)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            del main_texts, texts, images, labels, outputs, loss
            if total_batches % 10 == 0:
                self.clear_memory()

        # self.scheduler.step()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        for main_texts, texts, images, labels in self.val_loader:
            main_texts = main_texts.to(self.device, non_blocking=True)
            texts = texts.to(self.device, non_blocking=True)
            images = images.to(self.device, non_blocking=True)
            images = images.squeeze(1)  # Removes the dimension with size 1
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(main_texts, texts, images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            del main_texts, texts, images, labels, outputs, loss, preds

        self.clear_memory()

        avg_loss = total_loss / len(self.val_loader.dataset)
        metrics = self.calculate_metrics(all_preds, all_labels)

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
        # try:
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate()

            print(f"\nEpoch {epoch + 1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")
            self.print_metrics(val_metrics, "Validation")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

                # if torch.cuda.is_available():
                #     print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # except Exception as e:
        #     print(f"Training interrupted: {str(e)}")
        #     if self.best_model_state is not None:
        #         torch.save(self.best_model_state, "interrupted_model.pt")

    def test(self):
        if self.best_model_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_model_state.items()})
        test_loss, test_metrics = self.validate()
        print("\nBest Model Performance on Test Set:")
        self.print_metrics(test_metrics, "Test")
