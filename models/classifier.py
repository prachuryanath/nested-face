import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicLinearClassifier(nn.Module):
    # ... (Keep previous implementation or remove if unused) ...
    def __init__(self, embedding_dim=512, initial_classes=10):
        super(DynamicLinearClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = initial_classes
        self.fc = nn.Linear(embedding_dim, initial_classes)
        
    def forward(self, x):
        return self.fc(x)
    
    def expand(self, new_total_classes):
        if new_total_classes <= self.num_classes: return
        old_w, old_b = self.fc.weight.data, self.fc.bias.data
        self.fc = nn.Linear(self.embedding_dim, new_total_classes)
        self.fc.weight.data[:self.num_classes] = old_w
        self.fc.bias.data[:self.num_classes] = old_b
        self.num_classes = new_total_classes

class DynamicCosineClassifier(nn.Module):
    # ... (Keep previous implementation) ...
    def __init__(self, embedding_dim=512, initial_classes=10, scale=30.0):
        super(DynamicCosineClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = initial_classes
        self.scale = scale
        self.weight = nn.Parameter(torch.Tensor(initial_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        return cosine * self.scale

    def expand(self, new_total_classes):
        if new_total_classes <= self.num_classes: return
        old_w = self.weight.data
        self.weight = nn.Parameter(torch.Tensor(new_total_classes, self.embedding_dim))
        self.weight.data[:self.num_classes] = old_w
        nn.init.xavier_uniform_(self.weight.data[self.num_classes:])
        self.num_classes = new_total_classes

class DynamicArcFaceClassifier(nn.Module):
    """
    ArcFace Head with 'Industrial Strength' stability fixes.
    Prevents NaN explosions during incremental expansion.
    """
    def __init__(self, embedding_dim=512, initial_classes=10, s=30.0, m=0.50):
        super(DynamicArcFaceClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = initial_classes
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.Tensor(initial_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, labels=None):
        # 1. Safety: Normalize Input and Weights with epsilon
        # eps=1e-6 prevents div-by-zero if a vector is length 0
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-6)
        w_norm = F.normalize(self.weight, p=2, dim=1, eps=1e-6)
        
        cosine = F.linear(x_norm, w_norm)
        
        # 2. Safety: Strict Clamping
        # Floating point errors can make cosine 1.0000001 which breaks sqrt()
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None:
            return cosine * self.s
        
        # 3. Training Mode
        # cos(t + m) = cos(t)cos(m) - sin(t)sin(m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Robust Logic for theta > pi - m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Create One-Hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

    def expand(self, new_total_classes):
        if new_total_classes <= self.num_classes: 
            return
            
        print(f"ArcFace Expansion: {self.num_classes} -> {new_total_classes}")
        old_weight = self.weight.data
        self.weight = nn.Parameter(torch.Tensor(new_total_classes, self.embedding_dim))
        
        # Preserve old weights exactly
        self.weight.data[:self.num_classes] = old_weight
        
        # Initialize new weights safely
        # We use slightly smaller initialization to prevent massive gradients on first step
        nn.init.xavier_uniform_(self.weight.data[self.num_classes:])
        
        self.num_classes = new_total_classes