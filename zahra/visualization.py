"""
Visualization Module
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score
)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


# ============================================================================
# KNN VISUALIZATIONS
# ============================================================================

def plot_individual_knn_pipeline(
    y_test, y_pred, y_proba, name, best_params,
    acc, auc, f1, train_counts, test_counts
):
    """Individual dataset visualization with Pipeline info"""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'KNN Pipeline: {name}', fontsize=14, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Survived', 'Died']).plot(
        ax=axes[0,0], cmap='Blues', values_format='d'
    )
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].grid(False)
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0,1].plot(fpr, tpr, lw=2.5, label=f'AUC={auc:.3f}', color='darkorange')
    axes[0,1].plot([0,1], [0,1], 'k--', lw=1)
    axes[0,1].set_xlabel('FPR')
    axes[0,1].set_ylabel('TPR')
    axes[0,1].set_title('ROC Curve')
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)
    
    # 3. Pie Chart
    class_counts = pd.Series(y_test).value_counts()
    axes[0,2].pie(class_counts.values, 
                  labels=['Survived', 'Died'],
                  autopct='%1.1f%%',
                  colors=['skyblue', 'coral'],
                  startangle=90)
    axes[0,2].set_title('Test Distribution')
    
    # 4. Performance Metrics
    metrics = {'Accuracy': acc, 'F1-Score': f1, 'ROC-AUC': auc}
    axes[1,0].barh(list(metrics.keys()), list(metrics.values()),
                   color='steelblue', alpha=0.7, edgecolor='black')
    axes[1,0].set_xlim(0, 1.05)
    axes[1,0].set_xlabel('Score')
    axes[1,0].set_title('Performance Metrics')
    axes[1,0].grid(axis='x', alpha=0.3)
    for i, (metric, value) in enumerate(metrics.items()):
        axes[1,0].text(value + 0.02, i, f'{value:.3f}', 
                      va='center', fontweight='bold')
    
    # 5. Class Distribution
    x_pos = np.arange(2)
    width = 0.35
    train_survived = train_counts.get(0, 0)
    train_died = train_counts.get(1, 0)
    test_survived = test_counts.get(0, 0)
    test_died = test_counts.get(1, 0)
    
    axes[1,1].bar(x_pos - width/2, [train_survived, train_died], 
                  width, label='Train', alpha=0.8, color='skyblue')
    axes[1,1].bar(x_pos + width/2, [test_survived, test_died], 
                  width, label='Test', alpha=0.8, color='coral')
    axes[1,1].set_xlabel('Class')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_title('Class Distribution')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(['Survived', 'Died'])
    axes[1,1].legend()
    axes[1,1].grid(axis='y', alpha=0.3)
    
    # 6. Pipeline Info
    axes[1,2].axis('off')
    info_text = "Pipeline Configuration:\n\n"
    info_text += "Imputer → Selector → Scaler → KNN\n\n"
    info_text += f"k: {best_params['k']}\n"
    info_text += f"Method: {best_params['method']}\n"
    if best_params['detail']:
        info_text += f"{best_params['detail']}\n"
    info_text += f"\nAcc: {acc:.3f}\nAUC: {auc:.3f}"
    
    axes[1,2].text(0.1, 0.5, info_text, 
                  fontsize=10, family='monospace',
                  verticalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1,2].set_title('Model Info')
    
    plt.tight_layout()
    plt.show()


def plot_knn_comparison(all_results):
    """Comparison visualization for all KNN datasets"""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('KNN Pipeline: All Datasets Comparison', 
                 fontsize=16, fontweight='bold')
    
    names = [r['name'] for r in all_results]
    short_names = ['X0:Clin', 'X1:+A', 'X2:+B', 'X3:+Δ']
    
    # 1. Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    accs = [r['metrics']['accuracy'] for r in all_results]
    bars = ax1.bar(short_names, accs, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Accuracy')
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', fontweight='bold', fontsize=9)
    
    # 2. AUC
    ax2 = fig.add_subplot(gs[0, 1])
    aucs = [r['metrics']['roc_auc'] for r in all_results]
    bars = ax2.bar(short_names, aucs, color='darkorange', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('AUC', fontweight='bold')
    ax2.set_title('ROC-AUC')
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis='y', alpha=0.3)
    for bar, auc in zip(bars, aucs):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{auc:.3f}', ha='center', fontweight='bold', fontsize=9)
    
    # 3. F1
    ax3 = fig.add_subplot(gs[0, 2])
    f1s = [r['metrics']['f1'] for r in all_results]
    bars = ax3.bar(short_names, f1s, color='seagreen', alpha=0.7, edgecolor='black')
    ax3.set_ylabel('F1', fontweight='bold')
    ax3.set_title('F1-Score')
    ax3.set_ylim(0, 1.05)
    ax3.grid(axis='y', alpha=0.3)
    for bar, f1 in zip(bars, f1s):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{f1:.3f}', ha='center', fontweight='bold', fontsize=9)
    
    # 4. ROC Overlay
    ax4 = fig.add_subplot(gs[1, :2])
    colors = ['blue', 'green', 'red', 'purple']
    for i, result in enumerate(all_results):
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_proba'])
        ax4.plot(fpr, tpr, lw=2.5, color=colors[i], 
                label=f"{short_names[i]} (AUC={result['metrics']['roc_auc']:.3f})")
    ax4.plot([0,1], [0,1], 'k--', lw=1.5, label='Random')
    ax4.set_xlabel('FPR', fontweight='bold')
    ax4.set_ylabel('TPR', fontweight='bold')
    ax4.set_title('ROC Curves - All Datasets', fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    
    # 5. Pipeline Summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    ax5.text(0.5, 0.95, 'Pipeline Info', ha='center', fontsize=11, 
            fontweight='bold', transform=ax5.transAxes)
    
    for i, result in enumerate(all_results):
        y_pos = 0.80 - i*0.18
        params = result['best_params']
        text = f"{short_names[i]}: k={params['k']}, {params['method']}"
        ax5.text(0.05, y_pos, text, fontsize=8, 
                transform=ax5.transAxes, family='monospace')
    
    # 6. Heatmap
    ax6 = fig.add_subplot(gs[2, :])
    metrics_names = ['Acc', 'Bal-Acc', 'Prec', 'Rec', 'F1', 'AUC']
    heatmap_data = []
    for r in all_results:
        m = r['metrics']
        row = [m['accuracy'], m['balanced_acc'], m['precision'], 
               m['recall'], m['f1'], m['roc_auc']]
        heatmap_data.append(row)
    heatmap_data = np.array(heatmap_data)
    
    im = ax6.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax6.set_xticks(np.arange(len(metrics_names)))
    ax6.set_yticks(np.arange(len(short_names)))
    ax6.set_xticklabels(metrics_names)
    ax6.set_yticklabels(short_names)
    
    for i in range(len(short_names)):
        for j in range(len(metrics_names)):
            ax6.text(j, i, f'{heatmap_data[i, j]:.3f}',
                    ha="center", va="center", color="black", 
                    fontweight='bold', fontsize=9)
    
    ax6.set_title('All Metrics Heatmap', fontweight='bold', pad=10)
    cbar = plt.colorbar(im, ax=ax6, orientation='horizontal', pad=0.1, aspect=30)
    cbar.set_label('Score', fontweight='bold')
    
    plt.show()


# ============================================================================
# LOGISTIC REGRESSION VISUALIZATIONS
# ============================================================================

def create_logistic_comparison_plots(all_results: dict, datasets: dict) -> None:
    """Create comprehensive comparison plots for Logistic Regression - 6 panels like KNN"""
    
    print("\n DEBUG: Starting plot creation...")
    print(f"   Total results: {len(all_results)}")
    print(f"   Dataset keys: {list(datasets.keys())}")
    print(f"   Sample result keys: {list(all_results.keys())[:3]}")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('Logistic Regression Pipeline: All Datasets Comparison', 
                 fontsize=16, fontweight='bold')
    
    dataset_names = list(datasets.keys())
    
    # Get best result per dataset
    best_per_dataset = {}
    for ds_name in dataset_names:
        # Find all results for this dataset (keys start with dataset name)
        ds_results = {k: v for k, v in all_results.items() 
                     if k.startswith(ds_name)}
        
        print(f"\n   Dataset '{ds_name}': Found {len(ds_results)} results")
        
        if ds_results:
            # Find best by AUC
            best_combo = max(ds_results.items(), key=lambda x: x[1]['test_auc'])
            best_per_dataset[ds_name] = best_combo[1]
            print(f"      Best AUC: {best_combo[1]['test_auc']:.3f}")
        else:
            print(f"       No results found!")
    
    if not best_per_dataset:
        print("\n ERROR: No results found! Check dataset names and result keys.")
        return
    
    print(f"\n Found best results for {len(best_per_dataset)} datasets")
    
    # Extract metrics
    accuracies = []
    aucs = []
    f1s = []
    
    for name in dataset_names:
        if name in best_per_dataset:
            accuracies.append(best_per_dataset[name]['test_accuracy'])
            aucs.append(best_per_dataset[name]['test_auc'])
            f1s.append(best_per_dataset[name]['test_f1'])
        else:
            accuracies.append(0)
            aucs.append(0)
            f1s.append(0)
    
    print(f"   Accuracies: {accuracies}")
    print(f"   AUCs: {aucs}")
    print(f"   F1s: {f1s}")
    
    # Panel 1: Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(range(len(accuracies)), accuracies, 
                   color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xticks(range(len(dataset_names)))
    ax1.set_xticklabels(dataset_names, rotation=0)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy', fontweight='bold', fontsize=14)
    ax1.set_ylim([0, 1.05])
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, accuracies):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                    f'{val:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
    
    # Panel 2: ROC-AUC
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(range(len(aucs)), aucs, 
                   color='orange', edgecolor='black', alpha=0.7)
    ax2.set_xticks(range(len(dataset_names)))
    ax2.set_xticklabels(dataset_names, rotation=0)
    ax2.set_ylabel('AUC', fontsize=12)
    ax2.set_title('ROC-AUC', fontweight='bold', fontsize=14)
    ax2.set_ylim([0, 1.05])
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, aucs):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                    f'{val:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
    
    # Panel 3: F1-Score
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar(range(len(f1s)), f1s, 
                   color='green', edgecolor='black', alpha=0.7)
    ax3.set_xticks(range(len(dataset_names)))
    ax3.set_xticklabels(dataset_names, rotation=0)
    ax3.set_ylabel('F1', fontsize=12)
    ax3.set_title('F1-Score', fontweight='bold', fontsize=14)
    ax3.set_ylim([0, 1.05])
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, f1s):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                    f'{val:.3f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
    
    # Panel 4: ROC Curves
    ax4 = fig.add_subplot(gs[1, :2])
    colors = ['blue', 'green', 'red', 'purple']
    
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random', linewidth=1.5)
    
    for i, ds_name in enumerate(dataset_names):
        if ds_name in best_per_dataset:
            res = best_per_dataset[ds_name]
            if 'y_test_proba' in res and 'y_test' in res:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(res['y_test'], res['y_test_proba'])
                auc_val = res['test_auc']
                ax4.plot(fpr, tpr,
                        label=f"{ds_name} (AUC={auc_val:.3f})",
                        linewidth=2.5, color=colors[i % len(colors)])
    
    ax4.set_xlabel('False Positive Rate', fontsize=12)
    ax4.set_ylabel('True Positive Rate', fontsize=12)
    ax4.set_title('ROC Curves - All Datasets', fontweight='bold', fontsize=14)
    ax4.legend(fontsize=10, loc='lower right')
    ax4.grid(alpha=0.3)
    
    # Panel 5: Pipeline Info
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    ax5.text(0.5, 0.95, 'Pipeline Info', ha='center', fontsize=11, 
            fontweight='bold', transform=ax5.transAxes)
    
    for i, ds_name in enumerate(dataset_names):
        if ds_name in best_per_dataset:
            y_pos = 0.80 - i*0.18
            res = best_per_dataset[ds_name]
            penalty = res.get('penalty', 'unknown').upper()
            fs = res.get('fs_method', 'unknown').upper()
            C = res.get('best_params', {}).get('model__C', 'N/A')
            C_str = f"{C:.2f}" if isinstance(C, (int, float)) else str(C)
            
            text = f"{ds_name}: C={C_str}, {penalty}\n        {fs} FS"
            ax5.text(0.05, y_pos, text, fontsize=8, 
                    transform=ax5.transAxes, family='monospace')
    
    # Panel 6: Metrics Heatmap
    ax6 = fig.add_subplot(gs[2, :])
    
    metrics_data = []
    for name in dataset_names:
        if name in best_per_dataset:
            res = best_per_dataset[name]
            
            if 'y_test' in res and 'y_test_pred' in res:
                precision = precision_score(res['y_test'], res['y_test_pred'], 
                                          zero_division=0)
                recall = recall_score(res['y_test'], res['y_test_pred'], 
                                    zero_division=0)
            else:
                precision = 0.5
                recall = 0.5
            
            metrics_data.append([
                res['test_accuracy'],
                res['test_accuracy'],
                precision,
                recall,
                res['test_f1'],
                res['test_auc']
            ])
        else:
            metrics_data.append([0, 0, 0, 0, 0, 0])
    
    metrics_df = pd.DataFrame(
        metrics_data,
        index=dataset_names,
        columns=['Acc', 'Bal-Acc', 'Prec', 'Rec', 'F1', 'AUC']
    )
    
    sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, cbar_kws={'label': 'Score'},
                ax=ax6, linewidths=0.5)
    ax6.set_title('All Metrics Heatmap', fontweight='bold', fontsize=14)
    ax6.set_ylabel('')
    
    plt.tight_layout()
    plt.show()
    
    print("\n Comprehensive comparison plots created!")


def plot_logistic_comparison(best_results, dataset_names):
    """Plot comparison of best Logistic Regression results"""
    
    accuracies = [best_results[name]['test_accuracy'] for name in dataset_names]
    aucs = [best_results[name]['test_auc'] for name in dataset_names]
    f1s = [best_results[name]['test_f1'] for name in dataset_names]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Accuracy
    axes[0].bar(dataset_names, accuracies, color='steelblue', edgecolor='black')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy by Dataset', fontweight='bold', fontsize=13)
    axes[0].set_ylim([0, 1.05])
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, (x, y) in enumerate(zip(dataset_names, accuracies)):
        axes[0].text(i, y + 0.02, f'{y:.3f}', 
                    ha='center', fontweight='bold', fontsize=10)
    
    # AUC
    axes[1].bar(dataset_names, aucs, color='orange', edgecolor='black')
    axes[1].set_ylabel('AUC', fontsize=12)
    axes[1].set_title('ROC-AUC by Dataset', fontweight='bold', fontsize=13)
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, (x, y) in enumerate(zip(dataset_names, aucs)):
        axes[1].text(i, y + 0.02, f'{y:.3f}', 
                    ha='center', fontweight='bold', fontsize=10)
    
    # F1
    axes[2].bar(dataset_names, f1s, color='green', edgecolor='black')
    axes[2].set_ylabel('F1-Score', fontsize=12)
    axes[2].set_title('F1-Score by Dataset', fontweight='bold', fontsize=13)
    axes[2].set_ylim([0, 1.05])
    axes[2].grid(axis='y', alpha=0.3)
    
    for i, (x, y) in enumerate(zip(dataset_names, f1s)):
        axes[2].text(i, y + 0.02, f'{y:.3f}', 
                    ha='center', fontweight='bold', fontsize=10)
    
    plt.suptitle('Logistic Regression: Best Results per Dataset', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_penalty_comparison(all_results, datasets_list):
    """Compare different penalty types across datasets"""
    
    penalties = ['none', 'l1', 'l2', 'elasticnet']
    penalty_performance = {p: {'auc': [], 'acc': [], 'f1': []} for p in penalties}
    
    for ds_name in datasets_list:
        for penalty in penalties:
            combo_name = f"{ds_name}_{penalty}_none"
            
            if combo_name in all_results:
                res = all_results[combo_name]
                penalty_performance[penalty]['auc'].append(res['test_auc'])
                penalty_performance[penalty]['acc'].append(res['test_accuracy'])
                penalty_performance[penalty]['f1'].append(res['test_f1'])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    x = np.arange(len(datasets_list))
    width = 0.2
    colors = {'none': '#3498db', 'l1': '#e74c3c', 'l2': '#2ecc71', 'elasticnet': '#f39c12'}
    
    # AUC
    for i, penalty in enumerate(penalties):
        offset = (i - 1.5) * width
        axes[0].bar(x + offset, penalty_performance[penalty]['auc'], 
                   width, label=penalty.upper(), color=colors[penalty], alpha=0.8)
    
    axes[0].set_xlabel('Dataset', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('ROC-AUC', fontsize=12, fontweight='bold')
    axes[0].set_title('AUC by Penalty Type', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets_list)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 1])
    
    # Accuracy
    for i, penalty in enumerate(penalties):
        offset = (i - 1.5) * width
        axes[1].bar(x + offset, penalty_performance[penalty]['acc'], 
                   width, label=penalty.upper(), color=colors[penalty], alpha=0.8)
    
    axes[1].set_xlabel('Dataset', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_title('Accuracy by Penalty Type', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets_list)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # F1
    for i, penalty in enumerate(penalties):
        offset = (i - 1.5) * width
        axes[2].bar(x + offset, penalty_performance[penalty]['f1'], 
                   width, label=penalty.upper(), color=colors[penalty], alpha=0.8)
    
    axes[2].set_xlabel('Dataset', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    axes[2].set_title('F1-Score by Penalty Type', fontsize=13, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(datasets_list)
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].set_ylim([0, 1])
    
    plt.suptitle('Regularization Comparison: Performance Across Datasets', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_penalty_heatmap(all_results, datasets_list):
    """Create heatmap of penalty performance"""
    
    penalties = ['none', 'l1', 'l2', 'elasticnet']
    auc_matrix = np.zeros((len(penalties), len(datasets_list)))
    
    for i, penalty in enumerate(penalties):
        for j, ds_name in enumerate(datasets_list):
            combo_name = f"{ds_name}_{penalty}_none"
            if combo_name in all_results:
                auc_matrix[i, j] = all_results[combo_name]['test_auc']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(auc_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=datasets_list, yticklabels=[p.upper() for p in penalties],
                cbar_kws={'label': 'AUC'}, vmin=0.4, vmax=0.8, ax=ax,
                linewidths=1, linecolor='white')
    
    ax.set_title('ROC-AUC: Regularization Methods × Datasets', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Regularization', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_feature_selection_impact(all_results, datasets_dict):
    """Plot feature selection impact per dataset"""
    
    datasets_list = list(datasets_dict.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, ds_name in enumerate(datasets_list):
        ds_results = {k: v for k, v in all_results.items() 
                      if k.startswith(ds_name)}
        
        models = []
        aucs = []
        colors_list = []
        
        for combo_name, res in ds_results.items():
            parts = combo_name.split('_')
            penalty = parts[1] if len(parts) > 1 else '?'
            fs = parts[2] if len(parts) > 2 else '?'
            
            label = f"{penalty.upper()}\n{'FS' if fs == 'l1' else 'No FS'}"
            models.append(label)
            aucs.append(res['test_auc'])
            colors_list.append('#2ecc71' if fs == 'l1' else '#3498db')
        
        axes[idx].barh(models, aucs, color=colors_list, alpha=0.8, edgecolor='black')
        axes[idx].set_xlabel('ROC-AUC', fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{ds_name}: {datasets_dict[ds_name][1]}', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlim([0, 1])
        axes[idx].grid(axis='x', alpha=0.3)
        
        for i, (model, auc) in enumerate(zip(models, aucs)):
            axes[idx].text(auc + 0.02, i, f'{auc:.3f}', 
                          va='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('Feature Selection Impact: All Models per Dataset', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

print(" Visualization module loaded successfully!")