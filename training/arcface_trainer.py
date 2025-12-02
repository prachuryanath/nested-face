import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from models.backbone import ResNetFaceEmbedder
from models.classifier import DynamicArcFaceClassifier
from core.continuum import ContinuumMemory
from core.deep_optimizer import GradientModulator, NestedOptimizer

class NestedTrainer:
    def __init__(self, stream_generator, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stream = stream_generator
        
        print(f"Initializing Nested Trainer (Robust) on {self.device}...")
        
        self.backbone = ResNetFaceEmbedder().to(self.device)
        self.classifier = DynamicArcFaceClassifier(initial_classes=10).to(self.device) 
        
        self.memory = ContinuumMemory(capacity=2000)
        self.modulator = GradientModulator().to(self.device)
        self._init_modulator_bias()

        # Lower LR for stability
        self.model_opt = NestedOptimizer(
            list(self.backbone.parameters()) + list(self.classifier.parameters()),
            modulator_model=self.modulator,
            lr=0.005, 
            momentum=0.9
        )
        
        self.meta_opt = optim.Adam(self.modulator.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'acc': [], 'forgetting': []}

    def _init_modulator_bias(self):
        for m in self.modulator.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.0)
        self.modulator.net[-2].bias.data.fill_(3.0) 

    def train_task(self, task_id, epochs=5):
        print(f"\n>>> STARTING TASK {task_id} <<<")
        
        total_classes_so_far = sum([len(t) for t in self.stream.task_splits[:task_id+1]])
        self.classifier.expand(total_classes_so_far)
        self.classifier.to(self.device)
        
        self.model_opt = NestedOptimizer(
            list(self.backbone.parameters()) + list(self.classifier.parameters()),
            modulator_model=self.modulator,
            lr=0.005,
            momentum=0.9
        )

        train_loader = self.stream.get_task_loader(task_id, train=True, batch_size=32)
        
        self.backbone.train()
        self.classifier.train()
        self.modulator.train()
        
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Task {task_id} | Epoch {epoch+1}/{epochs}")
            
            for imgs_curr, lbls_curr in pbar:
                imgs_curr, lbls_curr = imgs_curr.to(self.device), lbls_curr.to(self.device)
                
                # A. Memory Retrieval
                imgs_mem, lbls_mem, _ = self.memory.sample(batch_size=32)
                
                # B. Forward Pass
                feats_curr = self.backbone(imgs_curr)
                logits_curr = self.classifier(feats_curr, labels=lbls_curr) 
                loss_curr = self.criterion(logits_curr, lbls_curr)

                loss_mem = torch.tensor(0.0).to(self.device)
                if imgs_mem is not None:
                    imgs_mem, lbls_mem = imgs_mem.to(self.device), lbls_mem.to(self.device)
                    feats_mem = self.backbone(imgs_mem)
                    logits_mem = self.classifier(feats_mem, labels=lbls_mem)
                    loss_mem = self.criterion(logits_mem, lbls_mem)

                # C. Optimization with Safety Checks
                self.model_opt.zero_grad()
                total_loss = loss_curr + 1.0 * loss_mem 
                
                # --- CRITICAL FIX: Circuit Breaker ---
                if torch.isnan(total_loss):
                    print(f" [WARNING] NaN detected in Loss! Skipping batch. (New: {loss_curr.item()}, Mem: {loss_mem.item()})")
                    self.model_opt.zero_grad() # Clear gradients
                    continue # Skip this step entirely
                # -------------------------------------

                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=5.0)
                
                self.model_opt.step() 
                
                # E. Update Memory
                with torch.no_grad():
                    # Check for NaNs in clean logits too
                    clean_logits = self.classifier(feats_curr, labels=None)
                    if not torch.isnan(clean_logits).any():
                        loss_per_sample = F.cross_entropy(clean_logits, lbls_curr, reduction='none')
                        self.memory.add(imgs_curr, lbls_curr, loss_per_sample)
                
                pbar.set_postfix({'L_New': f"{loss_curr.item():.3f}", 'L_Mem': f"{loss_mem.item():.3f}"})

    def evaluate(self, up_to_task_id):
        print(f"\n--- Evaluation after Task {up_to_task_id} ---")
        self.backbone.eval()
        self.classifier.eval()
        
        accuracies = []
        for t in range(up_to_task_id + 1):
            test_loader = self.stream.get_task_loader(t, train=False, batch_size=32)
            correct = 0
            total = 0
            
            with torch.no_grad():
                for imgs, lbls in test_loader:
                    imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                    feats = self.backbone(imgs)
                    logits = self.classifier(feats, labels=None)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == lbls).sum().item()
                    total += lbls.size(0)
            
            acc = 100 * correct / total
            accuracies.append(acc)
            print(f"Task {t} Accuracy: {acc:.3f}%")
            
        avg_acc = np.mean(accuracies)
        print(f"Average Accuracy: {avg_acc:.3f}%")
        self.history['acc'].append(avg_acc)
        return avg_acc