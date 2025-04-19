'''
The provided code defines functions to evaluate the performance of a machine learning model 
by calculating key metrics and visualizing them using precision-recall and ROC curves.'''
'''
precision_score: Measures the proportion of positive predictions that are correct
recall_score:Measures the proportion of actual positives correctly identified
f1_score(y_test, y_pred): The harmonic mean of precision and recall, balancing the two
roc_auc_score:Computes the Area Under the Receiver Operating Characteristic curve, 
indicating the model’s ability to distinguish classes
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve
)
def evaluate_model_performance(y_test, y_pred, y_prob):
    """
    Prints key evaluation metrics.
    """
    print("\nModel Evaluation:")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_test, y_prob):.4f}")
def plot_precision_recall_curve(y_test, y_prob):
    """
    Plots the precision-recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, marker='.')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_prob):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_curve.png', bbox_inches='tight')
    plt.show()
