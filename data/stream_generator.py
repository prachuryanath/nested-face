import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

class IncrementalFaceDataset(Dataset):
    """
    Wrapper to handle subsets of LFW for specific tasks.
    Remaps global class IDs to incremental local IDs (0..N).
    """
    def __init__(self, full_dataset, indices, class_mapping, transform=None):
        self.full_dataset = full_dataset
        self.indices = indices
        self.transform = transform
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, global_target = self.full_dataset[real_idx]
        
        # Apply transformation
        if self.transform:
            img = self.transform(img)
            
        # CRITICAL FIX: Remap global label (e.g., 88) to incremental label (e.g., 5)
        local_target = self.class_mapping[global_target]
            
        return img, local_target

class FaceStreamGenerator:
    def __init__(self, root_dir='./lfw-deepfunneled', n_tasks=5, min_faces_per_person=20):
        self.root_dir = root_dir
        self.n_tasks = n_tasks
        self.min_faces = min_faces_per_person
        
        # Standard Face Transformations
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f"--- Initializing Stream from: {self.root_dir} ---")
        self.prepare_data()

    def prepare_data(self):
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")

        print("Scanning local directory structure...")
        self.dataset = datasets.ImageFolder(root=self.root_dir, transform=None)
        
        targets = np.array(self.dataset.targets)
        classes = np.unique(targets)
        
        # Filter classes
        valid_classes = []
        for c in classes:
            if np.sum(targets == c) >= self.min_faces:
                valid_classes.append(c)
        
        self.valid_classes = np.array(valid_classes)
        print(f"Filtered down to {len(self.valid_classes)} identities.")
        
        if len(self.valid_classes) < self.n_tasks:
            raise ValueError(f"Not enough classes. Found {len(self.valid_classes)}, need {self.n_tasks}.")

        # Shuffle classes
        np.random.seed(42)
        np.random.shuffle(self.valid_classes)
        
        # Split classes into tasks
        self.task_splits = np.array_split(self.valid_classes, self.n_tasks)
        
        # --- CRITICAL FIX: Build Global -> Incremental Mapping ---
        self.global_to_incremental = {}
        current_label_counter = 0
        
        self.task_indices = []
        
        for task_id, class_group in enumerate(self.task_splits):
            indices = []
            
            # For every class in this task...
            for global_c in class_group:
                # 1. Map Global ID -> New Incremental ID
                self.global_to_incremental[global_c] = current_label_counter
                current_label_counter += 1
                
                # 2. Collect image indices
                c_indices = np.where(targets == global_c)[0]
                indices.extend(c_indices)
                
            self.task_indices.append(np.array(indices))
            print(f"Task {task_id}: {len(class_group)} classes (IDs {current_label_counter-len(class_group)} to {current_label_counter-1})")

    def get_task_loader(self, task_id, batch_size=32, train=True):
        if task_id >= self.n_tasks: raise ValueError("Invalid Task ID")
            
        all_indices = self.task_indices[task_id]
        all_targets = [self.dataset.targets[i] for i in all_indices]
        
        train_idx, test_idx = train_test_split(
            all_indices, test_size=0.2, stratify=all_targets, random_state=42
        )
        
        final_indices = train_idx if train else test_idx
        
        # Pass the mapping to the dataset
        task_dataset = IncrementalFaceDataset(
            self.dataset, 
            final_indices, 
            class_mapping=self.global_to_incremental, # <--- The Fix
            transform=self.transform
        )
        
        return DataLoader(task_dataset, batch_size=batch_size, shuffle=train, num_workers=0) # Set workers=0 for debugging safety

    def get_cumulative_test_loader(self, up_to_task, batch_size=32):
        cumulative_indices = []
        for t in range(up_to_task + 1):
            all_indices = self.task_indices[t]
            all_targets = [self.dataset.targets[i] for i in all_indices]
            _, test_idx = train_test_split(all_indices, test_size=0.2, stratify=all_targets, random_state=42)
            cumulative_indices.extend(test_idx)
            
        combined_dataset = IncrementalFaceDataset(
            self.dataset, 
            cumulative_indices, 
            class_mapping=self.global_to_incremental,
            transform=self.transform
        )
        return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, num_workers=0)