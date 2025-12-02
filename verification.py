import torch
import torch.nn as nn
import numpy as np
import os
import sys
from unittest.mock import MagicMock

# Add current directory to path so we can import modules
sys.path.append(os.getcwd())

# Import your modules
try:
    from models.backbone import ResNetFaceEmbedder
    from models.classifier import DynamicLinearClassifier
    from data.stream_generator import FaceStreamGenerator
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import modules. {e}")
    print("Ensure you have created the folder structure: models/ and data/")
    sys.exit(1)

def test_phase_1_data_stream():
    print("\n--- Testing Phase 1: Data Stream Logic ---")
    
    # 1. Mock the LFW Dataset to avoid downloading 200MB+
    # We create a fake dataset with 100 identities, 20 images each
    mock_dataset = MagicMock()
    mock_dataset.targets = []
    num_identities = 100
    images_per_id = 20
    
    for i in range(num_identities):
        mock_dataset.targets.extend([i] * images_per_id)
    
    mock_dataset.__len__.return_value = len(mock_dataset.targets)
    # Return random tensor for __getitem__
    mock_dataset.__getitem__.return_value = (torch.randn(3, 112, 112), 0)

    # 2. Patch the LFWPeople class in the module temporarily
    import data.stream_generator
    original_lfw = data.stream_generator.LFWPeople
    data.stream_generator.LFWPeople = MagicMock(return_value=mock_dataset)
    
    try:
        # Initialize Generator with 5 tasks
        stream = FaceStreamGenerator(n_tasks=5, min_faces_per_person=10)
        
        # Check Task Splitting
        assert len(stream.task_splits) == 5, "Failed to split into 5 tasks"
        
        # Verify disjoint identities
        task_0_ids = set(stream.task_splits[0])
        task_1_ids = set(stream.task_splits[1])
        intersection = task_0_ids.intersection(task_1_ids)
        assert len(intersection) == 0, f"Critical: Data Leakage! Task 0 and 1 share identities: {intersection}"
        
        print("✅ Data Stream: Task splitting logic is correct (Disjoint sets verified).")
        
        # Check DataLoader
        loader = stream.get_task_loader(0, batch_size=4)
        print("✅ Data Stream: DataLoader creation successful.")
        
    except Exception as e:
        print(f"❌ Phase 1 Failed: {e}")
        raise e
    finally:
        # Restore original class
        data.stream_generator.LFWPeople = original_lfw

def test_phase_2_models():
    print("\n--- Testing Phase 2: Subject Models ---")
    
    batch_size = 4
    embedding_dim = 512
    initial_classes = 10
    
    # 1. Test Backbone
    backbone = ResNetFaceEmbedder(embedding_dim=embedding_dim, pretrained=False)
    dummy_input = torch.randn(batch_size, 3, 112, 112)
    
    emb = backbone(dummy_input)
    assert emb.shape == (batch_size, embedding_dim), f"Backbone output mismatch. Expected {(batch_size, embedding_dim)}, got {emb.shape}"
    print("✅ Backbone: Forward pass shape correct.")
    
    # 2. Test Classifier & Expansion
    classifier = DynamicLinearClassifier(embedding_dim=embedding_dim, initial_classes=initial_classes)
    logits = classifier(emb)
    assert logits.shape == (batch_size, initial_classes), "Classifier output shape mismatch."
    
    # Capture old weights to verify they don't change
    old_weights = classifier.fc.weight.data.clone()
    
    # Expand Model
    new_classes = 20
    classifier.expand(new_classes)
    
    # Verify new shape
    new_logits = classifier(emb)
    assert new_logits.shape == (batch_size, new_classes), f"Expanded output mismatch. Expected {(batch_size, new_classes)}, got {new_logits.shape}"
    
    # Verify Old Weights Preservation (CRITICAL for Incremental Learning)
    current_weights = classifier.fc.weight.data
    # Compare only the slice corresponding to old classes
    diff = torch.sum(torch.abs(current_weights[:initial_classes] - old_weights))
    
    if diff < 1e-6:
        print("✅ Classifier: Expansion successful & Old weights preserved (Plasticity/Stability check passed).")
    else:
        print(f"❌ Classifier: Expansion corrupted old weights! Diff: {diff}")
        raise ValueError("Weights not preserved.")

if __name__ == "__main__":
    try:
        test_phase_1_data_stream()
        test_phase_2_models()
        print("\n🎉 ALL CHECKS PASSED: Phase 1 & 2 are ready for Nested Learning integration.")
    except Exception as e:
        print(f"\n🛑 VERIFICATION FAILED: {e}")