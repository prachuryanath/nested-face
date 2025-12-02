import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Import Subject Models
from models.backbone import ResNetFaceEmbedder
from models.classifier import DynamicCosineClassifier

class BaselineTrainer:
    """
    Standard Fine-Tuning Trainer (The 'Lower Bound').
    - No Continuum Memory (Replay)
    - No Deep Optimizer (Meta-Learning)
    - Just standard SGD on the incoming stream.
    
    Expected Outcome: Catastrophic Forgetting.
    """
    def __init__(self, stream_generator, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stream = stream_generator
        
        print(f"Initializing BASELINE Trainer (Naive Fine-Tuning) on {self.device}...")
        
        # Same Architecture as Nested Model for fair comparison
        self.backbone = ResNetFaceEmbedder().to(self.device)
        self.classifier = DynamicCosineClassifier(initial_classes=10).to(self.device)
        
        # Standard Optimizer (No Nested/Meta logic)
        self.optimizer = optim.SGD(
            list(self.backbone.parameters()) + list(self.classifier.parameters()),
            lr=0.01,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'acc': [], 'forgetting': []}

    def train_task(self, task_id, epochs=5):
        print(f"\n>>> [BASELINE] STARTING TASK {task_id} <<<")
        
        # 1. Expand Classifier
        total_classes = sum([len(t) for t in self.stream.task_splits[:task_id+1]])
        self.classifier.expand(total_classes)
        self.classifier.to(self.device)
        
        # Re-init optimizer for new params
        self.optimizer = optim.SGD(
            list(self.backbone.parameters()) + list(self.classifier.parameters()),
            lr=0.01, 
            momentum=0.9,
            weight_decay=5e-4
        )

        # 2. Get Data (No Memory Buffer here!)
        train_loader = self.stream.get_task_loader(task_id, train=True, batch_size=32)
        
        self.backbone.train()
        self.classifier.train()
        
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Task {task_id} | Epoch {epoch+1}/{epochs}")
            
            for imgs, lbls in pbar:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                
                # Standard Update
                self.optimizer.zero_grad()
                
                feats = self.backbone(imgs)
                logits = self.classifier(feats)
                loss = self.criterion(logits, lbls)
                
                loss.backward()
                self.optimizer.step()
                
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

    def evaluate(self, up_to_task_id):
        self.backbone.eval()
        self.classifier.eval()
        
        accuracies = []
        print(f"\n--- Baseline Evaluation after Task {up_to_task_id} ---")
        
        for t in range(up_to_task_id + 1):
            test_loader = self.stream.get_task_loader(t, train=False, batch_size=32)
            correct = 0
            total = 0
            
            with torch.no_grad():
                for imgs, lbls in test_loader:
                    imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                    feats = self.backbone(imgs)
                    logits = self.classifier(feats)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == lbls).sum().item()
                    total += lbls.size(0)
            
            acc = 100 * correct / total
            accuracies.append(acc)
            print(f"Task {t} Accuracy: {acc:.3f}%")
            
        avg_acc = np.mean(accuracies)
        print(f"Average Accuracy: {avg_acc:.3f}%")
        return avg_acc