import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class GradientModulator(nn.Module):
    """
    The 'Deep' part of the Deep Optimizer.
    Input: [Gradient, Parameter_State, Momentum_State]
    Output: [Gating_Value (0 to 1)]
    
    It learns to predict: "Should I allow this weight to change given the current conflict?"
    """
    def __init__(self, hidden_dim=64):
        super(GradientModulator, self).__init__()
        # Input features: 3 (Grad, Param, Momentum_Buffer)
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Gate: 0 = Freeze, 1 = Update
        )
        
    def forward(self, grad, param, momentum):
        # We process element-wise. 
        # Stack inputs: Shape [N, 3] where N is number of parameters
        x = torch.stack([grad, param, momentum], dim=-1)
        return self.net(x)

class NestedOptimizer(optim.Optimizer):
    """
    Custom PyTorch Optimizer implementing Nested Learning.
    Wraps standard SGD momentum logic but applies the GradientModulator gate.
    """
    def __init__(self, params, modulator_model, lr=0.01, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super(NestedOptimizer, self).__init__(params, defaults)
        self.modulator = modulator_model
        
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step using the Meta-Learner.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                d_p = p.grad
                
                # 1. Get/Init Momentum Buffer
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = state['momentum_buffer']
                    # Standard Momentum update
                    buf.mul_(mu).add_(d_p)
                
                # 2. Prepare Inputs for Modulator
                # We flatten tensors to [N] to feed into the element-wise MLP
                flat_grad = d_p.view(-1)
                flat_param = p.data.view(-1)
                flat_mom = buf.view(-1)
                
                # 3. Predict Gating Mask (The "Nested" Step)
                # Note: For efficiency in research code, we might chunk this if models are huge.
                # For ResNet18 (11M params), direct pass might OOM on small GPUs. 
                # We use a random subset or chunking in production. 
                # Here we apply a simple heuristic optimization:
                # Only apply modulator to the Classifier Head (most sensitive to forgetting)
                # and last block of ResNet.
                
                # For this implementation, we apply a simplified scalar modulation 
                # if tensor is too large, or full modulation if small.
                
                if flat_grad.numel() > 100000:
                    # Fallback to standard SGD for massive layers to save time/compute 
                    # (Or implement chunking here)
                    gate = 1.0 
                else:
                    # Run the Meta-Net
                    # Ensure modulator is on same device
                    if next(self.modulator.parameters()).device != flat_grad.device:
                        self.modulator.to(flat_grad.device)
                        
                    gate = self.modulator(flat_grad, flat_param, flat_mom)
                    gate = gate.view_as(d_p)

                # 4. Apply Update
                # New Parameter = Old - LR * Momentum * Gate
                # Gate < 1.0 reduces plasticity (prevents forgetting)
                # Gate near 1.0 allows full learning
                p.data.add_(buf * gate, alpha=-lr)

        return loss