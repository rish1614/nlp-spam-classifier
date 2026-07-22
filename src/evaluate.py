# src/evaluate.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Prints accuracy and a detailed classification report.
    Returns a dictionary of core metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    
    print(f"--- Evaluation for {model_name} ---")
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(report)
    
    return acc

def plot_confusion_matrix(y_true, y_pred, filepath=None):
    """
    Generates and optionally saves a heatmap of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if filepath:
        plt.savefig(filepath)
        print(f"Confusion matrix saved to {filepath}")
    else:
        plt.show()
