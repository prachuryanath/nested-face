import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import os
import json

sys.path.append(os.getcwd())

from training.trainer_baseline import BaselineTrainer
from training.trainer_strong import StrongBaselineTrainer 
from training.trainer_der import DERPlusPlusTrainer
from training.trainer_joint import JointTrainer # <--- NEW IMPORT
from data.stream_generator import FaceStreamGenerator

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_comparison(nested, baseline, strong, der, joint, filename="final_comparison_full.png"):
    tasks = list(range(len(baseline['average_accuracy'])))
    plt.figure(figsize=(12, 7))
    
    # 5. Joint / Upper Bound (Black Dashed)
    if joint:
        plt.plot(tasks, joint['average_accuracy'], 'k--', linewidth=2, label='Joint Training (Upper Bound)')

    # 4. Nested Learning (Green)
    if nested:
        plt.plot(tasks, nested['average_accuracy'], 'o-', color='green', linewidth=3, label='Nested Learning (Ours)')

    # 3. DER++ (Orange)
    if der:
        plt.plot(tasks, der['average_accuracy'], '^-', color='orange', linewidth=2, label='DER++ (SOTA)')
        
    # 2. iCaRL (Blue)
    if strong:
        plt.plot(tasks, strong['average_accuracy'], 's-', color='blue', alpha=0.7, label='iCaRL (Distillation)')

    # 1. Naive Baseline (Red)
    plt.plot(tasks, baseline['average_accuracy'], 'x--', color='red', alpha=0.4, label='Naive Fine-Tuning')

    plt.title("Nested Learning vs SOTA & Upper Bound", fontsize=16)
    plt.xlabel("Task ID", fontsize=12)
    plt.ylabel("Avg Accuracy (%)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

def run():
    print(">>> RUNNING FULL BENCHMARK SUITE <<<")
    set_seed(42)
    N_TASKS = 5
    stream = FaceStreamGenerator(root_dir='./lfw-deepfunneled', n_tasks=N_TASKS, min_faces_per_person=15)
    
    # A. Joint Training (Upper Bound)
    print("\n--- Running Joint Training (Upper Bound) ---")
    joint_trainer = JointTrainer(stream)
    joint_res = {'average_accuracy': []}
    for t in range(N_TASKS):
        joint_trainer.train_task(t, epochs=5) # Joint usually needs more epochs to converge on larger data
        joint_res['average_accuracy'].append(joint_trainer.evaluate(t))

    # B. DER++
    print("\n--- Running DER++ (SOTA) ---")
    der_trainer = DERPlusPlusTrainer(stream)
    der_res = {'average_accuracy': []}
    for t in range(N_TASKS):
        der_trainer.train_task(t, epochs=3)
        der_res['average_accuracy'].append(der_trainer.evaluate(t))

    # C. iCaRL
    print("\n--- Running iCaRL (Distillation) ---")
    strong_trainer = StrongBaselineTrainer(stream)
    strong_res = {'average_accuracy': []}
    for t in range(N_TASKS):
        strong_trainer.train_task(t, epochs=3)
        strong_res['average_accuracy'].append(strong_trainer.evaluate(t))

    # D. Naive
    print("\n--- Running Naive Baseline ---")
    naive_trainer = BaselineTrainer(stream)
    naive_res = {'average_accuracy': []}
    for t in range(N_TASKS):
        naive_trainer.train_task(t, epochs=3)
        naive_res['average_accuracy'].append(naive_trainer.evaluate(t))

    # E. Load Nested
    nested_res = None
    if os.path.exists("experiment_metricsj.json"):
        with open("experiment_metricsj.json", "r") as f:
            nested_res = json.load(f)
            
    plot_comparison(nested_res, naive_res, strong_res, der_res, joint_res)

if __name__ == "__main__":
    run()