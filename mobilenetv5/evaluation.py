"""
MobileNetV5 Evaluation Script
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from pathlib import Path
import os

from models import create_mobilenetv5
from training.dataset import create_dataloaders


class MobileNetV5Evaluator:
    """Evaluator for MobileNetV5 models"""
    
    def __init__(self, model_path: str, data_dir: str = '../bioast_dataset', 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.data_dir = data_dir
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        model_name = checkpoint['config']['model_name']
        self.model = create_mobilenetv5(model_name, num_classes=2, input_size=70)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Create dataloaders
        _, _, self.test_loader = create_dataloaders(data_dir, batch_size=32, image_size=70)
        
        print(f"Loaded model: {model_name}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
    
    def evaluate(self) -> dict:
        """Evaluate model on test set"""
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probability of positive class
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        return metrics
    
    def calculate_metrics(self, y_true: list, y_pred: list, y_prob: list) -> dict:
        """Calculate comprehensive evaluation metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # Medical-specific metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = recall_score(y_true, y_pred, average='binary')  # Same as recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # AUC
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': auc,
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'total_samples': len(y_true),
            'positive_samples': sum(y_true),
            'negative_samples': len(y_true) - sum(y_true)
        }
    
    def plot_confusion_matrix(self, y_true: list, y_pred: list, save_path: str = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def save_results(self, metrics: dict, output_dir: str = 'evaluation_results'):
        """Save evaluation results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to JSON serializable types
        metrics_json = self._convert_numpy_types(metrics)
        
        # Save metrics as JSON
        with open(output_path / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"Positive Samples: {metrics['positive_samples']}")
        print(f"Negative Samples: {metrics['negative_samples']}")
        print("\nConfusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"  True Negatives: {cm['true_negatives']}")
        print(f"  False Positives: {cm['false_positives']}")
        print(f"  False Negatives: {cm['false_negatives']}")
        print(f"  True Positives: {cm['true_positives']}")
        print("="*50)
        
        print(f"Results saved to {output_path}")


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate MobileNetV5')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='../bioast_dataset',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = MobileNetV5Evaluator(
        model_path=args.model_path,
        data_dir=args.data_dir
    )
    
    # Evaluate model
    metrics = evaluator.evaluate()
    
    # Save results
    evaluator.save_results(metrics, args.output_dir)


if __name__ == "__main__":
    main()