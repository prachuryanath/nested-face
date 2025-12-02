import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models.backbone import ResNetFaceEmbedder
from models.classifier import DynamicArcFaceClassifier
from data.stream_generator import IncrementalFaceDataset

class JointTrainer:
    """
    Joint Training (Upper Bound / Skyline).
    
    Strategy:
    At Task T, we construct a dataset containing ALL images from Task 0 to Task T.
    We train the model on this union. 
    This simulates 'Infinite Memory' and shows the theoretical maximum accuracy.
    """
    def __init__(self, stream_generator, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stream = stream_generator
        
        print(f"Initializing JOINT Trainer (Upper Bound) on {self.device}...")
        
        self.backbone = ResNetFaceEmbedder().to(self.device)
        # Use ArcFace for consistency with other strong baselines
        self.classifier = DynamicArcFaceClassifier(initial_classes=10).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()

    def get_cumulative_train_loader(self, up_to_task_id, batch_size=32):
        """
        Manually constructs a DataLoader containing Train Data from Task 0...up_to_task_id
        """
        cumulative_train_indices = []
        
        for t in range(up_to_task_id + 1):
            # 1. Get all indices for this task
            task_indices = self.stream.task_indices[t]
            task_targets = [self.stream.dataset.targets[i] for i in task_indices]
            
            # 2. Re-create the Train/Test split exactly as the StreamGenerator does
            # (Ensures we don't cheat by training on test set)
            train_idx, _ = train_test_split(
                task_indices, 
                test_size=0.2, 
                stratify=task_targets, 
                random_state=42
            )
            cumulative_train_indices.extend(train_idx)
            
        # 3. Create Unified Dataset
        joint_dataset = IncrementalFaceDataset(
            self.stream.dataset,
            cumulative_train_indices,
            class_mapping=self.stream.global_to_incremental,
            transform=self.stream.transform
        )
        
        return DataLoader(joint_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    def train_task(self, task_id, epochs=5):
        print(f"\n>>> [JOINT TRAINING] STARTING TASK {task_id} (Training on Tasks 0-{task_id}) <<<")
        
        # 1. Expand Classifier
        total_classes = sum([len(t) for t in self.stream.task_splits[:task_id+1]])
        self.classifier.expand(total_classes)
        self.classifier.to(self.device)
        
        # 2. Optimizer
        # In Joint Training, we often reset the optimizer or even the model 
        # to prove it can learn from scratch, but fine-tuning is also valid.
        # Here we fine-tune but with a fresh optimizer.
        self.optimizer = optim.SGD(
            list(self.backbone.parameters()) + list(self.classifier.parameters()),
            lr=0.01, momentum=0.9, weight_decay=5e-4
        )

        # 3. Get CUMULATIVE Loader
        train_loader = self.get_cumulative_train_loader(task_id, batch_size=32)
        
        self.backbone.train()
        self.classifier.train()
        
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Task {task_id} | Epoch {epoch+1}/{epochs}")
            
            for imgs, lbls in pbar:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                
                self.optimizer.zero_grad()
                
                feats = self.backbone(imgs)
                logits = self.classifier(feats, labels=lbls) # ArcFace Training
                loss = self.criterion(logits, lbls)
                
                if torch.isnan(loss):
                    print("NaN detected in Joint Training. Skipping.")
                    continue
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 5.0)
                self.optimizer.step()
                
                pbar.set_postfix({'Loss': f"{loss.item():.2f}"})

    def evaluate(self, up_to_task_id):
        self.backbone.eval()
        self.classifier.eval()
        accuracies = []
        # Evaluate on each task individually to match other reports
        for t in range(up_to_task_id + 1):
            test_loader = self.stream.get_task_loader(t, train=False, batch_size=32)
            correct = 0; total = 0
            with torch.no_grad():
                for imgs, lbls in test_loader:
                    imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                    # Inference (No labels passed to ArcFace)
                    logits = self.classifier(self.backbone(imgs), labels=None)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == lbls).sum().item()
                    total += lbls.size(0)
            acc = 100 * correct / total
            accuracies.append(acc)
            print(f"Task {t}: {acc:.2f}%")
        
        avg = np.mean(accuracies)
        print(f"Average Accuracy: {avg:.2f}%")
        return avg