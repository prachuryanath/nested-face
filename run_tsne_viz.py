import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.manifold import TSNE
import seaborn as sns
import sys
import os
import pandas as pd

sys.path.append(os.getcwd())

from training.trainer import NestedTrainer
from training.trainer_der import DERPlusPlusTrainer
from data.stream_generator import FaceStreamGenerator

# --- Configuration ---
N_TASKS = 5
EPOCHS = 10
CLASSES_PER_TASK_TO_PLOT = 5 # 5 Old + 5 New = 10 Clusters Total

def get_embeddings(trainer, stream, target_classes):
    trainer.backbone.eval()
    embeddings = []
    labels = []
    task_labels = [] 
    
    print("Extracting embeddings for visualization...")
    cumulative_loader = stream.get_cumulative_test_loader(up_to_task=N_TASKS-1, batch_size=32)
    
    with torch.no_grad():
        for imgs, lbls in cumulative_loader:
            imgs = imgs.to(trainer.device)
            feats = trainer.backbone(imgs)
            # Normalize for Cosine Similarity / ArcFace context
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            
            feats = feats.cpu().numpy()
            lbls = lbls.numpy()
            
            for i in range(len(lbls)):
                label = lbls[i]
                if label in target_classes:
                    embeddings.append(feats[i])
                    labels.append(label)
                    
                    if label < len(stream.task_splits[0]):
                        task_labels.append("Task 0 (Old Identity)")
                    else:
                        task_labels.append("Task 4 (New Identity)")

    return np.array(embeddings), np.array(labels), np.array(task_labels)

def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """
    if x.size < 2 or y.size < 2:
        return
        
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using a special case to obtain the eigenvalues of this
    # 2D dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def train_and_extract(trainer_cls, stream, name):
    print(f"\n>>> TRAINING {name} FOR VISUALIZATION <<<")
    trainer = trainer_cls(stream)
    
    for t in range(N_TASKS):
        trainer.train_task(t, epochs=EPOCHS)
        # --- NEW: Print Metrics ---
        print(f"\n--- {name}: Performance after Task {t} ---")
        avg_acc = trainer.evaluate(t)
        print(f"[{name}] Task {t} Avg Accuracy: {avg_acc:.2f}%")
        print("-" * 40)
    
    task0_ids = list(range(0, CLASSES_PER_TASK_TO_PLOT))
    total_classes_before_t4 = sum([len(t) for t in stream.task_splits[:4]])
    task4_ids = list(range(total_classes_before_t4, total_classes_before_t4 + CLASSES_PER_TASK_TO_PLOT))
    target_classes = set(task0_ids + task4_ids)
    
    emb, lbl, tasks = get_embeddings(trainer, stream, target_classes)
    return emb, lbl, tasks

def plot_tsne(emb_der, lbl_der, task_der, emb_nest, lbl_nest, task_nest):
    print("\nComputing t-SNE projection...")
    
    # Setup Figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), constrained_layout=True)
    
    def run_plot(ax, embs, labels, tasks, title):
        if len(embs) == 0:
            ax.text(0.5, 0.5, "No Data", ha='center'); return

        # Tuned t-SNE for tighter clusters
        tsne = TSNE(n_components=2, perplexity=25, max_iter=1500, learning_rate=200, init='pca', random_state=42)
        tsne_results = tsne.fit_transform(embs)
        
        df = pd.DataFrame({
            'x': tsne_results[:, 0],
            'y': tsne_results[:, 1],
            'Identity': labels,
            'Task Group': tasks
        })
        
        # 1. Draw Points
        sns.scatterplot(
            data=df, x='x', y='y', 
            hue='Identity', 
            style='Task Group',
            palette='bright', # High contrast
            s=150, # Bigger points
            edgecolor='white', linewidth=1, # Separation
            alpha=0.9,
            ax=ax,
            legend=False # We add custom legend later
        )
        
        # 2. Draw Confidence Ellipses (The "Cluster Shape")
        unique_ids = np.unique(labels)
        colors = sns.color_palette('bright', len(unique_ids))
        
        for i, uid in enumerate(unique_ids):
            subset = df[df['Identity'] == uid]
            # Draw ellipse
            confidence_ellipse(subset['x'], subset['y'], ax, n_std=2.0, 
                               edgecolor=colors[i], facecolor=colors[i], alpha=0.15)
            # Add center label
            ax.text(subset['x'].mean(), subset['y'].mean(), str(uid), 
                    fontsize=10, fontweight='bold', color='black', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")

        return df # Return for legend creation

    # Run Plots
    df_der = run_plot(axes[0], emb_der, lbl_der, task_der, "Baseline (DER++)")
    df_nest = run_plot(axes[1], emb_nest, lbl_nest, task_nest, "Nested Learning (Ours)")
    
    # Create a Shared Legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels() # This might be empty due to legend=False
    
    # Custom Legend Logic
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Task 0 (Old)', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='X', color='w', label='Task 4 (New)', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], color='gray', alpha=0.3, lw=4, label='Identity Cluster (2σ)')
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=14, bbox_to_anchor=(0.5, -0.05))
    
    plt.savefig("tsne_comparison_polished.png", dpi=300, bbox_inches='tight')
    print("\n✅ Polished t-SNE Plot saved to 'tsne_comparison_polished.png'")

def run():
    stream = FaceStreamGenerator(root_dir='./synth', n_tasks=N_TASKS, min_faces_per_person=15)
    
    # Train & Extract
    emb_der, lbl_der, task_der = train_and_extract(DERPlusPlusTrainer, stream, "DER++")
    emb_nest, lbl_nest, task_nest = train_and_extract(NestedTrainer, stream, "Nested Learning")
    
    plot_tsne(emb_der, lbl_der, task_der, emb_nest, lbl_nest, task_nest)

if __name__ == "__main__":
    run()