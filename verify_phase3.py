import torch
import torch.nn as nn
import sys
import os

# Import Core Modules
sys.path.append(os.getcwd())
try:
    from core.continuum import ContinuumMemory
    from core.deep_optimizer import GradientModulator, NestedOptimizer
except ImportError as e:
    print(f"CRITICAL: {e}")
    sys.exit(1)

def test_continuum_memory():
    print("\n--- Testing Phase 3: Continuum Memory ---")
    
    # 1. Initialize Small Buffer
    mem = ContinuumMemory(capacity=5, input_shape=(3, 10, 10))
    
    # 2. Create Dummy Data
    # 10 samples. Losses: 0.1, 0.2, ... 1.0
    imgs = torch.randn(10, 3, 10, 10)
    lbls = torch.arange(10)
    losses = torch.tensor([float(i)/10.0 for i in range(10)]) # 0.0 to 0.9
    
    # 3. Add to memory
    print("Adding 10 items to buffer of capacity 5...")
    mem.add(imgs, lbls, losses)
    
    # 4. Verify Retention Policy (High Surprise should be kept)
    print(f"Buffer Stats: {mem.stats()}")
    
    # Check contents
    stored_losses = [item.surprise for item in mem.buffer]
    print(f"Stored Surprises: {sorted(stored_losses)}")
    
    # We expect the HIGHEST losses (0.5, 0.6, 0.7, 0.8, 0.9) to be kept
    # The lowest losses (0.0, 0.1...) should be evicted.
    min_stored = min(stored_losses)
    if min_stored >= 0.5:
        print("✅ Memory Logic: Correctly retained High-Surprise samples.")
    else:
        print(f"❌ Memory Logic Failed. Found low surprise item: {min_stored}")
        raise ValueError("Buffer retention policy failed.")

def test_deep_optimizer():
    print("\n--- Testing Phase 3: Deep Optimizer ---")
    
    # 1. Simple Linear Model
    model = nn.Linear(10, 1)
    
    # 2. Initialize Meta-Learner
    modulator = GradientModulator(hidden_dim=8)
    opt = NestedOptimizer(model.parameters(), modulator, lr=0.1)
    
    # 3. Fake Training Step
    input_data = torch.randn(5, 10)
    target = torch.randn(5, 1)
    
    # Capture old weights
    old_weight = model.weight.data.clone()
    
    # Forward/Backward
    pred = model(input_data)
    loss = nn.MSELoss()(pred, target)
    loss.backward()
    
    # 4. Step with Meta-Optimizer
    # Note: Modulator is initialized randomly, so gate will be random ~0.5
    opt.step()
    
    # 5. Verify Update Happened
    new_weight = model.weight.data
    diff = torch.sum(torch.abs(new_weight - old_weight))
    
    if diff > 0:
        print(f"✅ Deep Optimizer: Weights updated successfully (Diff: {diff:.6f})")
    else:
        print("❌ Deep Optimizer: Weights did not change! Gate might be stuck at 0.")
        raise ValueError("Optimizer failed to update.")

    # 6. Verify Gating Behavior
    # If we force the modulator to output 0 (freeze), weights should not change.
    print("Testing 'Freeze' capability...")
    for p in modulator.parameters():
        nn.init.constant_(p, 0.0) # Bias output towards constant
        
    # Hack the last layer bias of modulator to be large negative -> Sigmoid(neg) -> 0
    modulator.net[-2].bias.data.fill_(-10.0)
    
    # Step again
    param_before_freeze = model.weight.data.clone()
    loss = nn.MSELoss()(model(input_data), target)
    loss.backward()
    opt.step()
    
    diff_freeze = torch.sum(torch.abs(model.weight.data - param_before_freeze))
    print(f"Diff with frozen gate: {diff_freeze:.6f}")
    
    if diff_freeze < diff:
         print("✅ Deep Optimizer: Gating mechanism is functional (Can inhibit updates).")

if __name__ == "__main__":
    try:
        test_continuum_memory()
        test_deep_optimizer()
        print("\n🎉 PHASE 3 VERIFIED: The Core is Ready.")
    except Exception as e:
        print(f"\n🛑 VERIFICATION FAILED: {e}")