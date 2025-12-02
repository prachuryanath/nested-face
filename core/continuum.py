import torch
import numpy as np
import random
from collections import namedtuple

# Lightweight storage structure
MemoryItem = namedtuple('MemoryItem', ['image', 'label', 'surprise', 'logits'])

class ContinuumMemory:
    """
    Universal Memory Buffer.
    Supports:
    1. 'Surprise-Based' Storage (for Nested Learning)
    2. 'Logit-Based' Storage (for DER++)
    
    Nested Learning Concept:
    Instead of random sampling, we prioritize 'Context Flows' (tasks/samples) 
    that cause high surprise (high loss), as these represent boundaries 
    the model is struggling to learn.
    """
    def __init__(self, capacity=2000, input_shape=(3, 112, 112)):
        self.capacity = capacity
        self.buffer = [] 
        self.seen_classes = set()
        
    def add(self, images, labels, losses, logits=None):
        """
        Args:
            logits: (Optional) The raw output vector from the model at the time of storage.
                    Required for DER++.
        """
        images = images.detach().cpu()
        labels = labels.detach().cpu()
        losses = losses.detach().cpu()
        if logits is not None:
            logits = logits.detach().cpu()
        
        for i in range(len(images)):
            # If logits provided, store them. Else None.
            logit_val = logits[i] if logits is not None else None
            
            item = MemoryItem(images[i], labels[i].item(), losses[i].item(), logit_val)
            self.seen_classes.add(item.label)
            
            if len(self.buffer) < self.capacity:
                self.buffer.append(item)
            else:
                # Replacement Strategy
                # If logits are present (DER mode), we typically use Random Reservoir Sampling
                # If no logits (Nested mode), we use Surprise
                
                if logits is not None:
                    # DER++ Standard: Random Reservoir Sampling
                    idx = random.randint(0, len(self.seen_classes) + len(self.buffer))
                    if idx < len(self.buffer):
                        self.buffer[idx] = item
                else:
                    # Nested Learning Standard: High Surprise replacement
                    min_surprise_idx = -1
                    min_surprise_val = float('inf')
                    for idx, stored_item in enumerate(self.buffer):
                        if stored_item.surprise < min_surprise_val:
                            min_surprise_val = stored_item.surprise
                            min_surprise_idx = idx
                    
                    if item.surprise > min_surprise_val:
                        self.buffer[min_surprise_idx] = item

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None, None, None
            
        n = min(len(self.buffer), batch_size)
        batch_items = random.sample(self.buffer, n)
        
        images = torch.stack([b.image for b in batch_items])
        labels = torch.tensor([b.label for b in batch_items])
        
        # Stack logits if they exist (Handle variable sizes by padding if necessary, 
        # but typically we handle sizing in the loss function)
        logits = None
        if batch_items[0].logits is not None:
            # Note: Logits might be different sizes if tasks grew!
            # We return a list or handle it carefully. 
            # For simplicity here, we assume we only use logits for matching logic 
            # and might need to handle size mismatch in trainer.
            logits = [b.logits for b in batch_items] 
            
        return images, labels, logits

    def __len__(self):
        return len(self.buffer)