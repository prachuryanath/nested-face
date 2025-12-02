import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import os
import json
import time

# --- 1. Path Setup (Crucial for importing modules) ---
# Adds the current directory to Python path so we can import 'models', 'core', etc.
sys.path.append(os.getcwd())

try:
    from training.arcface_trainer import NestedTrainer
    from data.stream_generator import FaceStreamGenerator
except ImportError as e:
    print(f"CRITICAL ERROR: {e}")
    print("Run this script from the root directory: 'python run_experiment.py'")
    sys.exit(1)

# --- 2. Reproducibility ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- 3. Visualization ---
def plot_results(history, filename="forgetting_curve.png"):
    """
    Plots the accuracy trajectory over tasks.
    """
    tasks = list(range(len(history['acc'])))
    accs = history['acc']
    
    plt.figure(figsize=(10, 6))
    plt.plot(tasks, accs, marker='o', linewidth=2, label='Nested Learning (Ours)')
    
    plt.title("Incremental Learning Performance (LFW)", fontsize=14)
    plt.xlabel("Task ID (New Identities Added)", fontsize=12)
    plt.ylabel("Average Accuracy (%)", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(tasks)
    plt.legend()
    
    print(f"Saving plot to {filename}...")
    plt.savefig(filename)
    plt.close()

# --- 4. Main Execution ---
def run():
    print("========================================")
    print("   NESTED LEARNING: EXPERIMENT START    ")
    print("========================================")
    
    # A. Configuration
    SEED = 42
    N_TASKS = 5
    EPOCHS_PER_TASK = 5 # Increase this for better results (e.g., 20)
    MIN_FACES = 15      # Minimum images per identity to be included
    
    set_seed(SEED)
    
    # B. Initialize Data Stream
    print(f"\n[1/4] Initializing Data Stream (Split LFW)...")
    try:
        stream = FaceStreamGenerator(n_tasks=N_TASKS, min_faces_per_person=MIN_FACES)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # C. Initialize Trainer
    print(f"\n[2/4] Initializing Nested Trainer...")
    trainer = NestedTrainer(stream)
    
    # D. Training Loop
    print(f"\n[3/4] Starting Incremental Training ({N_TASKS} Tasks)...")
    start_time = time.time()
    
    results = {
        'task_accuracy': [],
        'average_accuracy': [],
        'forgetting': [] # Accuracy of Task 0 over time
    }
    
    for task_id in range(N_TASKS):
        print(f"\n>>> Training Phase: Task {task_id} <<<")
        
        # Train
        trainer.train_task(task_id, epochs=EPOCHS_PER_TASK)
        
        # Evaluate (Calculates Avg Acc over 0..task_id)
        avg_acc = trainer.evaluate(task_id)
        results['average_accuracy'].append(avg_acc)
        
        # Detailed Check: specifically check Task 0 to measure pure forgetting
        # (This requires peeking into trainer's internal eval, so we do a custom check)
        task0_loader = stream.get_task_loader(0, train=False, batch_size=32)
        correct = 0
        total = 0
        trainer.backbone.eval()
        trainer.classifier.eval()
        with torch.no_grad():
            for img, lbl in task0_loader:
                img, lbl = img.to(trainer.device), lbl.to(trainer.device)
                pred = trainer.classifier(trainer.backbone(img)).argmax(1)
                correct += (pred == lbl).sum().item()
                total += lbl.size(0)
        task0_acc = 100 * correct / total
        results['forgetting'].append(task0_acc)
        print(f"    Task 0 Retention: {task0_acc:.3f}%")

    total_time = time.time() - start_time
    print(f"\n[4/4] Experiment Complete. Time: {total_time/60:.2f} mins.")

    # E. Save Results
    # 1. Plot
    plot_results({'acc': results['average_accuracy']}, filename="arcface_result_curve.png")
    
    # 2. JSON
    with open("experiment_metricsj.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Metrics saved to experiment_metricsj.json")
    
    print("\nSUMMARY:")
    print(f"Final Average Accuracy: {results['average_accuracy'][-1]:.3f}%")
    print(f"Task 0 Final Retention: {results['forgetting'][-1]:.3f}%")

if __name__ == "__main__":
    run()