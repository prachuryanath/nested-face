import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from models.backbone import ResNetFaceEmbedder
from models.classifier import DynamicArcFaceClassifier
from core.continuum import ContinuumMemory

class DERPlusPlusTrainer:
    """
    DER++ with Consistency Regularization for Face Recognition.
    
    Improvements:
    1. Consistency Reg: Applies transforms (Flip/Jitter) to memory images.
       Forces student(Transform(x)) ~= Teacher(x).
    2. Stability: Handles variable logit sizes during expansion.
    """
    def __init__(self, stream_generator, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stream = stream_generator
        
        print(f"Initializing DER++ Trainer (Robust) on {self.device}...")
        
        self.backbone = ResNetFaceEmbedder().to(self.device)
        self.classifier = DynamicArcFaceClassifier(initial_classes=10).to(self.device)
        
        # Memory stores Logits
        self.memory = ContinuumMemory(capacity=2000)
        
        self.optimizer = optim.SGD(
            list(self.backbone.parameters()) + list(self.classifier.parameters()),
            lr=0.01, momentum=0.9, weight_decay=5e-4
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Hyperparameters
        self.alpha = 0.5 # Logit Matching weight
        self.beta = 0.5  # Buffer Label weight

    def _augment(self, images):
        """
        Applies random augmentation to tensor batch for Consistency Regularization.
        Face Recognition relies heavily on alignment, so we only use Flips/Color,
        avoiding heavy crops/rotations that break alignment.
        """
        # 1. Random Horizontal Flip
        if torch.rand(1) < 0.5:
            images = torch.flip(images, dims=[3])
            
        # 2. Slight Pixel Intensity Jitter (Brightness)
        noise = torch.rand(images.size(0), 1, 1, 1, device=images.device) * 0.2 + 0.9
        images = images * noise
        
        return images

    def train_task(self, task_id, epochs=5):
        print(f"\n>>> [DER++] STARTING TASK {task_id} <<<")
        
        total_classes = sum([len(t) for t in self.stream.task_splits[:task_id+1]])
        self.classifier.expand(total_classes)
        self.classifier.to(self.device)
        
        self.optimizer = optim.SGD(
            list(self.backbone.parameters()) + list(self.classifier.parameters()),
            lr=0.01, momentum=0.9
        )

        train_loader = self.stream.get_task_loader(task_id, train=True, batch_size=32)
        
        self.backbone.train()
        self.classifier.train()
        
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Task {task_id} | Epoch {epoch+1}/{epochs}")
            
            for imgs, lbls in pbar:
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)
                
                self.optimizer.zero_grad()
                
                # --- 1. Forward Current Data ---
                feats = self.backbone(imgs)
                # Training: Use Labels for Margin
                logits_margin = self.classifier(feats, labels=lbls)
                loss_main = self.criterion(logits_margin, lbls)
                
                loss_der = torch.tensor(0.0).to(self.device)
                loss_buf = torch.tensor(0.0).to(self.device)

                # --- 2. Replay with Consistency Regularization ---
                # Unpack all 3 return values
                buf_imgs, buf_lbls, buf_logits_list = self.memory.sample(batch_size=32)
                
                if buf_imgs is not None:
                    buf_imgs, buf_lbls = buf_imgs.to(self.device), buf_lbls.to(self.device)
                    
                    # Apply Consistency Augmentation
                    # We want Student(Aug(x)) to match Teacher(x)
                    aug_buf_imgs = self._augment(buf_imgs)
                    
                    buf_feats = self.backbone(aug_buf_imgs)
                    
                    # A. Buffer Cross Entropy (Replay)
                    buf_logits_margin = self.classifier(buf_feats, labels=buf_lbls)
                    loss_buf = self.criterion(buf_logits_margin, buf_lbls)
                    
                    # B. Logit Matching (Dark Experience)
                    # Get raw logits (no margin) for matching
                    buf_logits_curr_raw = self.classifier(buf_feats, labels=None)
                    
                    mse_loss_accum = 0.0
                    valid_items = 0
                    
                    for i, stored_logit in enumerate(buf_logits_list):
                        if stored_logit is not None:
                            stored_logit = stored_logit.to(self.device)
                            
                            # Slice current logits to match stored size
                            n_old = stored_logit.shape[0]
                            curr_logit_slice = buf_logits_curr_raw[i, :n_old]
                            
                            # MSE Loss: Force consistency despite augmentation
                            mse_loss_accum += F.mse_loss(curr_logit_slice, stored_logit)
                            valid_items += 1
                    
                    if valid_items > 0:
                        loss_der = mse_loss_accum / valid_items

                # Total Loss
                loss = loss_main + (self.alpha * loss_der) + (self.beta * loss_buf)
                
                if torch.isnan(loss):
                    print("NaN detected in DER++ Loss. Skipping.")
                    continue
                    
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.backbone.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 5.0)
                
                self.optimizer.step()
                
                # --- 3. Update Memory ---
                with torch.no_grad():
                    # Store CLEAN (non-augmented) logits for future matching
                    # Use raw logits (no margin) for targets
                    raw_logits = self.classifier(feats, labels=None)
                    dummy_loss = torch.zeros(len(imgs))
                    self.memory.add(imgs, lbls, dummy_loss, logits=raw_logits)

                pbar.set_postfix({
                    'L_Main': f"{loss_main.item():.2f}", 
                    'L_DER': f"{loss_der.item():.2f}"
                })

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
                    feats=self.backbone(imgs)
                    logits = self.classifier(feats, labels=None)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == lbls).sum().item()
                    total += lbls.size(0)
            acc = 100 * correct / total
            accuracies.append(acc)
            print(f"Task {t}: {acc:.2f}%")
        return np.mean(accuracies)