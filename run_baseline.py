import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import os
import json

sys.path.append(os.getcwd())

from training.trainer_baseline import BaselineTrainer
from training.trainer_strong import StrongBaselineTrainer # <--- IMPORT NEW TRAINER
from data.stream_generator import FaceStreamGenerator

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_comparison(nested, baseline, strong, filename="final_comparison.png"):
    tasks = list(range(len(baseline['average_accuracy'])))
    plt.figure(figsize=(10, 6))
    
    # 1. Naive Baseline (Red)
    plt.plot(tasks, baseline['average_accuracy'], 'x--', color='red', label='Naive Fine-Tuning')
    
    # 2. Strong Baseline (Blue)
    if strong:
        plt.plot(tasks, strong['average_accuracy'], 's-', color='blue', label='Strong Baseline (Distillation)')

    # 3. Nested Learning (Green)
    if nested:
        plt.plot(tasks, nested['average_accuracy'], 'o-', color='green', linewidth=2, label='Nested Learning (Ours)')

    plt.title("Nested Learning vs Baselines", fontsize=14)
    plt.xlabel("Task ID")
    plt.ylabel("Avg Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

def run():
    print(">>> RUNNING BASELINE EXPERIMENTS <<<")
    set_seed(42)
    N_TASKS = 50
    stream = FaceStreamGenerator(root_dir='./synth', n_tasks=N_TASKS, min_faces_per_person=40)
    
    # --- A. Run Naive Baseline ---
    print("\n--- Running Naive Baseline ---")
    naive_trainer = BaselineTrainer(stream)
    naive_res = {'average_accuracy': []}
    for t in range(N_TASKS):
        naive_trainer.train_task(t, epochs=10) # Shorter epochs for speed
        naive_res['average_accuracy'].append(naive_trainer.evaluate(t))
        
    # --- B. Run Strong Baseline ---
    print("\n--- Running Strong Baseline (Replay + Distillation) ---")
    strong_trainer = StrongBaselineTrainer(stream)
    strong_res = {'average_accuracy': []}
    for t in range(N_TASKS):
        strong_trainer.train_task(t, epochs=10)
        strong_res['average_accuracy'].append(strong_trainer.evaluate(t))
        
    # --- C. Load Nested Results ---
    nested_res = None
    if os.path.exists("experiment_metricsL.json"):
        with open("experiment_metricsL.json", "r") as f:
            nested_res = json.load(f)
            
    plot_comparison(nested_res, naive_res, strong_res)

if __name__ == "__main__":
    run()