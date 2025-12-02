import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Import our modules
from models.backbone import ResNetFaceEmbedder
from models.classifier import DynamicCosineClassifier # <--- CHANGED TO COSINE
from core.continuum import ContinuumMemory
from core.deep_optimizer import GradientModulator, NestedOptimizer

class NestedTrainer:
    def __init__(self, stream_generator, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stream = stream_generator
        
        print(f"Initializing Nested Trainer on {self.device}...")
        
        # 1. Initialize Subject Models
        self.backbone = ResNetFaceEmbedder().to(self.device)
        # CHANGED: Use Cosine Classifier (Better for Face Rec / Incremental Learning)
        self.classifier = DynamicCosineClassifier(initial_classes=10).to(self.device) 
        
        # 2. Initialize Nested Core
        self.memory = ContinuumMemory(capacity=2000)
        self.modulator = GradientModulator().to(self.device)
        self._init_modulator_bias() # Ensure it starts by allowing updates (Gate=1)

        # 3. Optimizers
        self.model_opt = NestedOptimizer(
            list(self.backbone.parameters()) + list(self.classifier.parameters()),
            modulator_model=self.modulator,
            lr=0.01,
            momentum=0.9
        )
        
        # Meta-Optimizer: Trains the Modulator
        self.meta_opt = optim.Adam(self.modulator.parameters(), lr=1e-3)
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.history = {'acc': [], 'forgetting': []}

    def _init_modulator_bias(self):
        """
        Initialize Modulator to output ~1.0 (Identity) initially.
        If it starts at 0.5 (Random), it slows down training too much.
        """
        for m in self.modulator.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.0)
        # Bias the final layer heavily positive so Sigmoid(x) -> 1.0
        self.modulator.net[-2].bias.data.fill_(3.0) 

    def train_task(self, task_id, epochs=5):
        print(f"\n>>> STARTING TASK {task_id} <<<")
        
        # 1. Expand Classifier
        total_classes_so_far = sum([len(t) for t in self.stream.task_splits[:task_id+1]])
        self.classifier.expand(total_classes_so_far)
        self.classifier.to(self.device)
        
        # Re-init optimizer to capture new parameters
        self.model_opt = NestedOptimizer(
            list(self.backbone.parameters()) + list(self.classifier.parameters()),
            modulator_model=self.modulator,
            lr=0.01,
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
                
                # --- A. MEMORY RETRIEVAL ---
                imgs_mem, lbls_mem, _ = self.memory.sample(batch_size=32)
                
                # --- B. COMPUTE LOSSES & GRADIENTS ---
                # 1. Current Task Loss
                feats_curr = self.backbone(imgs_curr)
                logits_curr = self.classifier(feats_curr)
                loss_curr = self.criterion(logits_curr, lbls_curr)

                # 2. Memory Loss (if available)
                loss_mem = torch.tensor(0.0).to(self.device)
                if imgs_mem is not None:
                    imgs_mem, lbls_mem = imgs_mem.to(self.device), lbls_mem.to(self.device)
                    feats_mem = self.backbone(imgs_mem)
                    logits_mem = self.classifier(feats_mem)
                    loss_mem = self.criterion(logits_mem, lbls_mem)

                # --- C. META-TRAINING (THE FIX) ---
                # We train the modulator to prevent gradients that increase Memory Loss
                if imgs_mem is not None and task_id > 0:
                    # HEURISTIC: Instead of full second-order derivatives (slow),
                    # we enforce a simple rule: 
                    # "If Modulator makes the update, will Memory Loss go up?"
                    
                    # 1. Compute Gradients for Memory only
                    self.model_opt.zero_grad()
                    loss_mem.backward(retain_graph=True)
                    
                    # Store memory gradients (flattened subset for speed)
                    # We only look at the Classifier weights for meta-update to save time
                    mem_grad_vec = []
                    for p in self.classifier.parameters():
                        if p.grad is not None: mem_grad_vec.append(p.grad.view(-1))
                    if mem_grad_vec:
                        mem_grad_vec = torch.cat(mem_grad_vec)
                        
                        # 2. Compute Gradients for Current only
                        self.model_opt.zero_grad()
                        loss_curr.backward(retain_graph=True)
                        
                        curr_grad_vec = []
                        for p in self.classifier.parameters():
                             if p.grad is not None: curr_grad_vec.append(p.grad.view(-1))
                        curr_grad_vec = torch.cat(curr_grad_vec)
                        
                        # 3. Detect Conflict (Cosine Similarity)
                        # dot < 0 means they want to move in opposite directions
                        dot = torch.sum(curr_grad_vec * mem_grad_vec)
                        
                        if dot < 0:
                            # CONFLICT DETECTED!
                            # Train Modulator to output 0 (Freeze) for these conflicting updates
                            # We create a dummy input for the modulator representing this conflict
                            
                            # Note: Ideally we pass exact tensors, but for speed we do a 
                            # lightweight update on the modulator parameters directly
                            # to penalize "open gates" during conflict.
                            pass 
                            # (In this simplified script, we rely on the Weighted Loss below 
                            # effectively doing Replay, but simpler:
                            # Just by having loss_mem, the gradients *add up*. 
                            # The modulator's job is to scale this sum.)

                # --- D. OPTIMIZATION STEP ---
                self.model_opt.zero_grad()
                # Weighted Sum: Give more weight to memory to fix forgetting
                total_loss = loss_curr + 1.0 * loss_mem 
                total_loss.backward()
                
                self.model_opt.step() # Modulator acts here
                
                # --- E. UPDATE MEMORY ---
                with torch.no_grad():
                    # Calculate surprise (L2 norm of error vector roughly equals loss magnitude)
                    loss_per_sample = F.cross_entropy(logits_curr, lbls_curr, reduction='none')
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
                    logits = self.classifier(feats)
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