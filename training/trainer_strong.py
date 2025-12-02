import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import numpy as np
from tqdm import tqdm

from models.backbone import ResNetFaceEmbedder
from models.classifier import DynamicCosineClassifier
from core.continuum import ContinuumMemory

class StrongBaselineTrainer:
    """
    Implements 'Replay + Distillation' (similar to iCaRL).
    Fixed: Snapshots model BEFORE expansion.
    """
    def __init__(self, stream_generator, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stream = stream_generator
        
        self.backbone = ResNetFaceEmbedder().to(self.device)
        self.classifier = DynamicCosineClassifier(initial_classes=10).to(self.device)
        
        self.old_backbone = None
        self.old_classifier = None
        
        self.memory = ContinuumMemory(capacity=2000)
        
        self.optimizer = optim.SGD(
            list(self.backbone.parameters()) + list(self.classifier.parameters()),
            lr=0.01, momentum=0.9, weight_decay=5e-4
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_task(self, task_id, epochs=5):
        print(f"\n>>> [STRONG BASELINE] STARTING TASK {task_id} <<<")
        
        # 0. Calculate Class Counts
        total_classes = sum([len(t) for t in self.stream.task_splits[:task_id+1]])
        # Critical: Check current size before expansion
        old_classes = self.classifier.num_classes 
        
        # 1. Snapshot Old Model (The Teacher) -- MOVED BEFORE EXPANSION
        if task_id > 0:
            print("Snapshotting Old Teacher Model...")
            self.old_backbone = copy.deepcopy(self.backbone)
            self.old_backbone.eval()
            self.old_classifier = copy.deepcopy(self.classifier)
            self.old_classifier.eval()
            
            # Freeze teacher to save memory/safety
            for p in self.old_backbone.parameters(): p.requires_grad = False
            for p in self.old_classifier.parameters(): p.requires_grad = False

        # 2. Expand Classifier (Student)
        self.classifier.expand(total_classes)
        self.classifier.to(self.device)
        
        # Re-init optimizer
        self.optimizer = optim.SGD(
            list(self.backbone.parameters()) + list(self.classifier.parameters()),
            lr=0.01, momentum=0.9, weight_decay=5e-4
        )

        train_loader = self.stream.get_task_loader(task_id, train=True, batch_size=32)
        
        self.backbone.train()
        self.classifier.train()
        
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Task {task_id} | Epoch {epoch+1}/{epochs}")
            
            for imgs, lbls in pbar:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                
                # Retrieve Memory
                imgs_mem, lbls_mem, _ = self.memory.sample(batch_size=32)
                
                if imgs_mem is not None:
                    imgs_mem, lbls_mem = imgs_mem.to(self.device), lbls_mem.to(self.device)
                    combined_imgs = torch.cat([imgs, imgs_mem])
                    combined_lbls = torch.cat([lbls, lbls_mem])
                else:
                    combined_imgs = imgs
                    combined_lbls = lbls
                
                self.optimizer.zero_grad()
                
                # --- Forward Pass ---
                feats = self.backbone(combined_imgs)
                logits = self.classifier(feats)
                
                # 1. Classification Loss
                loss_cls = self.criterion(logits, combined_lbls)
                
                # 2. Distillation Loss
                loss_dist = torch.tensor(0.0).to(self.device)
                
                if task_id > 0:
                    with torch.no_grad():
                        old_feats = self.old_backbone(combined_imgs)
                        old_logits = self.old_classifier(old_feats)
                    
                    # Distill only on the shared subset of classes
                    # Student's logits for [0..old_classes]
                    new_logits_old_classes = logits[:, :old_classes]
                    
                    # Match shapes: [Batch, 20] vs [Batch, 20]
                    T = 2.0
                    loss_dist = nn.KLDivLoss(reduction='batchmean')(
                        F.log_softmax(new_logits_old_classes / T, dim=1),
                        F.softmax(old_logits / T, dim=1)
                    ) * (T * T)

                total_loss = loss_cls + loss_dist
                
                total_loss.backward()
                self.optimizer.step()
                
                # Update Memory
                with torch.no_grad():
                    dummy_loss = torch.ones(len(imgs)).to(self.device)
                    self.memory.add(imgs, lbls, dummy_loss)
                
                pbar.set_postfix({'Cls': f"{loss_cls.item():.3f}", 'Dist': f"{loss_dist.item():.3f}"})

    def evaluate(self, up_to_task_id):
        self.backbone.eval()
        self.classifier.eval()
        accuracies = []
        for t in range(up_to_task_id + 1):
            test_loader = self.stream.get_task_loader(t, train=False, batch_size=32)
            correct = 0; total = 0
            with torch.no_grad():
                for imgs, lbls in test_loader:
                    imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                    preds = self.classifier(self.backbone(imgs)).argmax(1)
                    correct += (preds == lbls).sum().item()
                    total += lbls.size(0)
            acc = 100 * correct / total
            accuracies.append(acc)
            print(f"Task {t}: {acc:.3f}%")
        return np.mean(accuracies)