"""
Enhanced Error Sample Analysis for MIC MobileNetV3 Models.

This script analyzes misclassified samples to identify patterns and characteristics
that can guide further model improvements.
"""

import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.enhanced_mic_mobilenetv3 import create_enhanced_mic_mobilenetv3
from training.dataset import BioastDataset

class ErrorSampleAnalyzer:
    """Comprehensive error sample analysis for MIC MobileNetV3."""
    
    def __init__(self, model_path: str, model_type: str = 'enhanced'):
        self.model_path = model_path
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Create data loaders
        self.test_loader = self._create_test_loader()
        
        # Analysis results
        self.error_samples = []
        self.feature_embeddings = []
        self.predictions = []
        self.ground_truths = []
        self.confidence_scores = []
        
    def _load_model(self):
        """Load the trained model."""
        if self.model_type == 'enhanced':
            model = create_enhanced_mic_mobilenetv3(num_classes=2)
        else:
            # For comparison with original model
            from models.mic_mobilenetv3 import create_mic_mobilenetv3
            model = create_mic_mobilenetv3(num_classes=2)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        return model
    
    def _create_test_loader(self):
        """Create test data loader."""
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = BioastDataset('bioast_dataset', split='test', transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        return test_loader
    
    def extract_features_and_predictions(self):
        """Extract features and predictions for all test samples."""
        print("🔍 Extracting features and predictions from test set...")
        
        all_features = []
        all_predictions = []
        all_ground_truths = []
        all_confidence_scores = []
        all_image_paths = []
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(self.test_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                
                # Extract features (from global pooling layer)
                features = self.model.forward_features(data)
                pooled_features = self.model.global_pool(features).flatten(1)
                
                # Get predictions
                logits = outputs['classification']
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                confidence = torch.max(probabilities, dim=1)[0]
                
                # Store results
                all_features.append(pooled_features.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
                all_ground_truths.append(targets.cpu().numpy())
                all_confidence_scores.append(confidence.cpu().numpy())
                
                print(f"   Processed batch {batch_idx + 1}/{len(self.test_loader)}")
        
        # Concatenate all results
        self.feature_embeddings = np.concatenate(all_features, axis=0)
        self.predictions = np.concatenate(all_predictions, axis=0)
        self.ground_truths = np.concatenate(all_ground_truths, axis=0)
        self.confidence_scores = np.concatenate(all_confidence_scores, axis=0)
        
        print(f"✅ Extracted features for {len(self.predictions)} samples")
        
    def identify_error_samples(self):
        """Identify and categorize error samples."""
        print("🔍 Identifying error samples...")
        
        # Find misclassified samples
        errors = self.predictions != self.ground_truths
        error_indices = np.where(errors)[0]
        
        print(f"📊 Found {len(error_indices)} error samples out of {len(self.predictions)} total")
        print(f"   Error rate: {len(error_indices) / len(self.predictions) * 100:.2f}%")
        
        # Categorize errors
        false_positives = []  # Predicted positive, actually negative
        false_negatives = []  # Predicted negative, actually positive
        
        for idx in error_indices:
            sample_info = {
                'index': idx,
                'predicted': self.predictions[idx],
                'ground_truth': self.ground_truths[idx],
                'confidence': self.confidence_scores[idx],
                'features': self.feature_embeddings[idx]
            }
            
            if self.predictions[idx] == 1 and self.ground_truths[idx] == 0:
                false_positives.append(sample_info)
            elif self.predictions[idx] == 0 and self.ground_truths[idx] == 1:
                false_negatives.append(sample_info)
        
        print(f"   False Positives (predicted positive, actually negative): {len(false_positives)}")
        print(f"   False Negatives (predicted negative, actually positive): {len(false_negatives)}")
        
        self.error_samples = {
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'all_errors': error_indices
        }
        
        return self.error_samples
    
    def analyze_confidence_patterns(self):
        """Analyze confidence score patterns."""
        print("🔍 Analyzing confidence patterns...")
        
        # Calculate confidence statistics
        correct_predictions = self.predictions == self.ground_truths
        
        correct_confidence = self.confidence_scores[correct_predictions]
        error_confidence = self.confidence_scores[~correct_predictions]
        
        confidence_analysis = {
            'correct_samples': {
                'mean_confidence': float(np.mean(correct_confidence)),
                'std_confidence': float(np.std(correct_confidence)),
                'min_confidence': float(np.min(correct_confidence)),
                'max_confidence': float(np.max(correct_confidence)),
                'count': len(correct_confidence)
            },
            'error_samples': {
                'mean_confidence': float(np.mean(error_confidence)),
                'std_confidence': float(np.std(error_confidence)),
                'min_confidence': float(np.min(error_confidence)),
                'max_confidence': float(np.max(error_confidence)),
                'count': len(error_confidence)
            }
        }
        
        # High confidence errors (most concerning)
        high_conf_threshold = 0.8
        high_conf_errors = error_confidence > high_conf_threshold
        
        print(f"📊 Confidence Analysis:")
        print(f"   Correct samples: {confidence_analysis['correct_samples']['mean_confidence']:.3f} ± {confidence_analysis['correct_samples']['std_confidence']:.3f}")
        print(f"   Error samples: {confidence_analysis['error_samples']['mean_confidence']:.3f} ± {confidence_analysis['error_samples']['std_confidence']:.3f}")
        print(f"   High-confidence errors (>{high_conf_threshold}): {np.sum(high_conf_errors)} ({np.sum(high_conf_errors)/len(error_confidence)*100:.1f}%)")
        
        return confidence_analysis
    
    def perform_feature_clustering(self):
        """Perform clustering analysis on features."""
        print("🔍 Performing feature clustering analysis...")
        
        # Dimensionality reduction for visualization
        print("   Running PCA...")
        pca = PCA(n_components=50)
        features_pca = pca.fit_transform(self.feature_embeddings)
        
        print("   Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_tsne = tsne.fit_transform(features_pca)
        
        # Clustering
        print("   Performing K-means clustering...")
        n_clusters = 6  # Based on error types and patterns
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.feature_embeddings)
        
        # Analyze error distribution across clusters
        error_mask = self.predictions != self.ground_truths
        cluster_analysis = {}
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_errors = np.sum(error_mask & cluster_mask)
            cluster_total = np.sum(cluster_mask)
            error_rate = cluster_errors / cluster_total if cluster_total > 0 else 0
            
            cluster_analysis[f'cluster_{cluster_id}'] = {
                'total_samples': int(cluster_total),
                'error_samples': int(cluster_errors),
                'error_rate': float(error_rate),
                'accuracy': float(1 - error_rate)
            }
        
        print(f"📊 Cluster Analysis:")
        for cluster_id, stats in cluster_analysis.items():
            print(f"   {cluster_id}: {stats['total_samples']} samples, {stats['error_rate']:.1%} error rate")
        
        return {
            'cluster_labels': cluster_labels,
            'features_tsne': features_tsne,
            'cluster_analysis': cluster_analysis,
            'pca_variance_ratio': pca.explained_variance_ratio_
        }
    
    def analyze_class_specific_errors(self):
        """Analyze errors specific to each class."""
        print("🔍 Analyzing class-specific error patterns...")
        
        class_analysis = {}
        
        for class_idx in [0, 1]:  # Negative and Positive
            class_name = 'negative' if class_idx == 0 else 'positive'
            
            # Samples that are actually this class
            actual_class_mask = self.ground_truths == class_idx
            actual_class_samples = np.sum(actual_class_mask)
            
            # Correctly predicted samples of this class
            correct_predictions = (self.predictions == class_idx) & actual_class_mask
            correct_count = np.sum(correct_predictions)
            
            # Incorrectly predicted samples
            incorrect_predictions = (self.predictions != class_idx) & actual_class_mask
            incorrect_count = np.sum(incorrect_predictions)
            
            # Confidence analysis for this class
            if actual_class_samples > 0:
                class_confidence = self.confidence_scores[actual_class_mask]
                correct_confidence = self.confidence_scores[correct_predictions]
                incorrect_confidence = self.confidence_scores[incorrect_predictions] if incorrect_count > 0 else []
                
                class_analysis[class_name] = {
                    'total_samples': int(actual_class_samples),
                    'correct_predictions': int(correct_count),
                    'incorrect_predictions': int(incorrect_count),
                    'accuracy': float(correct_count / actual_class_samples),
                    'mean_confidence': float(np.mean(class_confidence)),
                    'correct_mean_confidence': float(np.mean(correct_confidence)) if len(correct_confidence) > 0 else 0,
                    'incorrect_mean_confidence': float(np.mean(incorrect_confidence)) if len(incorrect_confidence) > 0 else 0
                }
        
        print(f"📊 Class-specific Analysis:")
        for class_name, stats in class_analysis.items():
            print(f"   {class_name.capitalize()}:")
            print(f"     Accuracy: {stats['accuracy']:.1%}")
            print(f"     Mean confidence: {stats['mean_confidence']:.3f}")
            print(f"     Correct predictions confidence: {stats['correct_mean_confidence']:.3f}")
            print(f"     Incorrect predictions confidence: {stats['incorrect_mean_confidence']:.3f}")
        
        return class_analysis
    
    def generate_improvement_recommendations(self, analyses: Dict):
        """Generate specific improvement recommendations based on error analysis."""
        print("🎯 Generating improvement recommendations...")
        
        recommendations = []
        
        # 1. Confidence-based recommendations
        conf_analysis = analyses['confidence']
        if conf_analysis['error_samples']['mean_confidence'] > 0.7:
            recommendations.append({
                'type': 'confidence_calibration',
                'priority': 'high',
                'issue': f"High confidence in errors ({conf_analysis['error_samples']['mean_confidence']:.3f})",
                'solution': "Implement confidence calibration techniques (temperature scaling, Platt scaling)",
                'implementation': [
                    "Add temperature scaling layer after final classification",
                    "Train calibration parameters on validation set",
                    "Apply label smoothing with higher alpha (0.2-0.3)",
                    "Use dropout during inference for uncertainty estimation"
                ]
            })
        
        # 2. Class imbalance recommendations
        class_analysis = analyses['class_specific']
        neg_acc = class_analysis['negative']['accuracy']
        pos_acc = class_analysis['positive']['accuracy']
        
        if abs(neg_acc - pos_acc) > 0.05:  # 5% difference
            worse_class = 'negative' if neg_acc < pos_acc else 'positive'
            better_class = 'positive' if neg_acc < pos_acc else 'negative'
            
            recommendations.append({
                'type': 'class_imbalance',
                'priority': 'medium',
                'issue': f"{worse_class.capitalize()} class has lower accuracy ({class_analysis[worse_class]['accuracy']:.1%} vs {class_analysis[better_class]['accuracy']:.1%})",
                'solution': "Implement advanced class balancing techniques",
                'implementation': [
                    f"Increase sampling weight for {worse_class} class",
                    "Use class-aware data augmentation",
                    "Implement cost-sensitive learning",
                    "Add class-specific regularization terms"
                ]
            })
        
        # 3. Feature representation recommendations
        cluster_analysis = analyses['clustering']['cluster_analysis']
        high_error_clusters = [k for k, v in cluster_analysis.items() if v['error_rate'] > 0.1]
        
        if len(high_error_clusters) > 0:
            recommendations.append({
                'type': 'feature_representation',
                'priority': 'medium',
                'issue': f"{len(high_error_clusters)} clusters have high error rates (>10%)",
                'solution': "Improve feature representation for problematic sample groups",
                'implementation': [
                    "Add more attention mechanisms (spatial, channel, self-attention)",
                    "Implement hard negative mining for difficult samples",
                    "Use contrastive learning to separate confusing classes",
                    "Add regularization to encourage diverse feature learning"
                ]
            })
        
        # 4. Architecture improvements
        error_rate = len(analyses['errors']['all_errors']) / len(self.predictions)
        if error_rate > 0.05:  # >5% error rate
            recommendations.append({
                'type': 'architecture',
                'priority': 'low',
                'issue': f"Overall error rate is {error_rate:.1%}",
                'solution': "Consider architectural improvements",
                'implementation': [
                    "Increase model capacity (wider or deeper networks)",
                    "Add residual connections for better gradient flow",
                    "Implement progressive training (curriculum learning)",
                    "Use ensemble methods for robust predictions"
                ]
            })
        
        # 5. Data augmentation recommendations
        fp_count = len(analyses['errors']['false_positives'])
        fn_count = len(analyses['errors']['false_negatives'])
        
        if fp_count > fn_count * 1.5:  # More false positives
            recommendations.append({
                'type': 'data_augmentation',
                'priority': 'medium',
                'issue': f"High false positive rate ({fp_count} vs {fn_count} false negatives)",
                'solution': "Enhance negative class representation",
                'implementation': [
                    "Add more air bubble simulation in negative samples",
                    "Increase optical noise augmentation",
                    "Implement negative mining for hard negatives",
                    "Use adversarial training with false positive examples"
                ]
            })
        elif fn_count > fp_count * 1.5:  # More false negatives
            recommendations.append({
                'type': 'data_augmentation',
                'priority': 'medium',
                'issue': f"High false negative rate ({fn_count} vs {fp_count} false positives)",
                'solution': "Enhance positive class representation",
                'implementation': [
                    "Add more bacterial growth simulation",
                    "Increase turbidity variation augmentation",
                    "Implement positive class oversampling",
                    "Use synthetic positive sample generation"
                ]
            })
        
        return recommendations
    
    def save_analysis_report(self, analyses: Dict, recommendations: List):
        """Save comprehensive analysis report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = {
            'metadata': {
                'timestamp': timestamp,
                'model_type': self.model_type,
                'model_path': self.model_path,
                'total_samples': len(self.predictions),
                'error_samples': len(analyses['errors']['all_errors']),
                'error_rate': len(analyses['errors']['all_errors']) / len(self.predictions)
            },
            'confidence_analysis': analyses['confidence'],
            'class_specific_analysis': analyses['class_specific'],
            'clustering_analysis': analyses['clustering']['cluster_analysis'],
            'error_breakdown': {
                'false_positives': len(analyses['errors']['false_positives']),
                'false_negatives': len(analyses['errors']['false_negatives'])
            },
            'improvement_recommendations': recommendations
        }
        
        # Save JSON report
        report_path = f"error_analysis_report_{self.model_type}_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Analysis report saved to: {report_path}")
        
        return report_path
    
    def run_complete_analysis(self):
        """Run the complete error analysis pipeline."""
        print("🚀 Starting comprehensive error sample analysis...")
        print("=" * 60)
        
        # Step 1: Extract features and predictions
        self.extract_features_and_predictions()
        
        # Step 2: Identify error samples
        errors = self.identify_error_samples()
        
        # Step 3: Analyze confidence patterns
        confidence_analysis = self.analyze_confidence_patterns()
        
        # Step 4: Perform clustering analysis
        clustering_analysis = self.perform_feature_clustering()
        
        # Step 5: Class-specific analysis
        class_analysis = self.analyze_class_specific_errors()
        
        # Compile all analyses
        all_analyses = {
            'errors': errors,
            'confidence': confidence_analysis,
            'clustering': clustering_analysis,
            'class_specific': class_analysis
        }
        
        # Step 6: Generate recommendations
        recommendations = self.generate_improvement_recommendations(all_analyses)
        
        # Step 7: Save report
        report_path = self.save_analysis_report(all_analyses, recommendations)
        
        print("\n" + "=" * 60)
        print("🎯 Analysis Summary:")
        print(f"   Total samples analyzed: {len(self.predictions)}")
        print(f"   Error samples found: {len(errors['all_errors'])}")
        print(f"   Error rate: {len(errors['all_errors']) / len(self.predictions) * 100:.2f}%")
        print(f"   False positives: {len(errors['false_positives'])}")
        print(f"   False negatives: {len(errors['false_negatives'])}")
        print(f"   Improvement recommendations: {len(recommendations)}")
        print(f"   Report saved to: {report_path}")
        print("=" * 60)
        
        return all_analyses, recommendations, report_path

def main():
    parser = argparse.ArgumentParser(description='Error Sample Analysis for MIC MobileNetV3')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='enhanced',
                        choices=['enhanced', 'original'],
                        help='Type of model (enhanced or original)')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = ErrorSampleAnalyzer(args.model_path, args.model_type)
    analyses, recommendations, report_path = analyzer.run_complete_analysis()
    
    # Print key recommendations
    print("\n🎯 Key Improvement Recommendations:")
    for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
        print(f"\n{i}. {rec['type'].replace('_', ' ').title()} ({rec['priority']} priority)")
        print(f"   Issue: {rec['issue']}")
        print(f"   Solution: {rec['solution']}")
        if rec['implementation']:
            print(f"   Implementation:")
            for impl in rec['implementation'][:2]:  # Show first 2 implementation steps
                print(f"     - {impl}")

if __name__ == "__main__":
    main()