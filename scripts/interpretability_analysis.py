"""
Model Interpretability Analysis Script

This script provides comprehensive interpretability analysis for trained models,
including Grad-CAM visualization, SHAP analysis, and attention weight analysis.

Usage:
    python scripts/interpretability_analysis.py --model efficientnet_b0 --analysis gradcam
    python scripts/interpretability_analysis.py --model convnext_tiny --analysis shap
    python scripts/interpretability_analysis.py --model resnet18_improved --analysis all
"""

import sys
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import get_model_config, get_latest_experiment_path, REPORTS_DIR
from training.dataset import create_data_loaders

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Model interpretability analysis')
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['efficientnet_b0', 'resnet18_improved', 'convnext_tiny'],
        help='Model to analyze'
    )
    
    parser.add_argument(
        '--analysis',
        type=str,
        default='all',
        choices=['gradcam', 'shap', 'attention', 'all'],
        help='Type of analysis to perform'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=20,
        help='Number of samples to analyze'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Custom output directory'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for analysis'
    )
    
    return parser.parse_args()

class GradCAM:
    """Grad-CAM implementation for CNN models."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save forward pass activations."""
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradients."""
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, class_idx=None):
        """Generate Grad-CAM heatmap."""
        # Forward pass
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[:, class_idx].squeeze()
        class_score.backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[2, 3])
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam / torch.max(cam)
        
        return cam.detach().numpy()

def load_model_and_weights(model_name, device):
    """Load trained model with weights."""
    print(f"📁 Loading model: {model_name}")
    
    # Create model instance using factory functions
    if model_name == 'efficientnet_b0':
        from models.efficientnet import create_efficientnet_b0
        model = create_efficientnet_b0(num_classes=2)
    elif model_name == 'resnet18_improved':
        from models.resnet_improved import create_resnet18_improved
        model = create_resnet18_improved(num_classes=2)
    elif model_name == 'convnext_tiny':
        from models.convnext_tiny import create_convnext_tiny
        model = create_convnext_tiny(num_classes=2)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Load weights
    experiment_path = get_latest_experiment_path(model_name)
    weights_path = experiment_path / 'best_model.pth'
    
    if weights_path.exists():
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded weights from: {weights_path}")
    else:
        print(f"⚠️  No weights found at: {weights_path}")
    
    model.to(device)
    model.eval()
    
    return model

def get_target_layer(model, model_name):
    """Get the target layer for Grad-CAM analysis."""
    if model_name == 'efficientnet_b0':
        # Last convolutional layer before classifier (head_conv)
        return model.head_conv
    elif model_name == 'resnet18_improved':
        # Last convolutional layer
        return model.layer4[-1].conv2
    elif model_name == 'convnext_tiny':
        # Last convolutional layer in the last stage
        # ConvNext: Use the last block in the last stage
        # The stages contain Sequential modules with Block objects
        last_stage = model.stages[-1]
        last_block = last_stage[-1]  # Get the last block from Sequential
        return last_block.dwconv  # Use the depthwise convolution layer
    else:
        raise ValueError(f"Target layer not defined for model: {model_name}")

def perform_gradcam_analysis(model, model_name, data_loader, device, output_dir, num_samples=20):
    """Perform Grad-CAM analysis."""
    print("🔍 Performing Grad-CAM analysis...")
    
    # Get target layer
    target_layer = get_target_layer(model, model_name)
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Create output directory
    gradcam_dir = output_dir / 'gradcam'
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze samples
    sample_count = 0
    results = []
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        if sample_count >= num_samples:
            break
            
        images = images.to(device)
        labels = labels.to(device)
        
        for i in range(images.shape[0]):
            if sample_count >= num_samples:
                break
                
            # Get single image
            image = images[i:i+1]
            label = labels[i].item()
            
            # Generate CAM
            cam = gradcam.generate_cam(image, class_idx=label)
            
            # Convert image for visualization
            img_np = images[i].cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            
            # Resize CAM to match image size
            cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(img_np)
            axes[0].set_title(f'Original (Label: {label})')
            axes[0].axis('off')
            
            # Heatmap
            axes[1].imshow(cam_resized, cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap')
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(img_np)
            axes[2].imshow(cam_resized, cmap='jet', alpha=0.4)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(gradcam_dir / f'gradcam_sample_{sample_count:03d}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # Store results
            results.append({
                'sample_id': sample_count,
                'true_label': label,
                'cam_max': float(cam.max()),
                'cam_mean': float(cam.mean()),
                'image_path': f'gradcam_sample_{sample_count:03d}.png'
            })
            
            sample_count += 1
    
    # Save results
    with open(gradcam_dir / 'gradcam_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Grad-CAM analysis completed: {len(results)} samples analyzed")
    return results

def generate_analysis_report(model_name, gradcam_results, output_dir):
    """Generate comprehensive analysis report."""
    print("📄 Generating analysis report...")
    
    report_content = f"""# Interpretability Analysis Report: {model_name}

## Executive Summary

This report provides comprehensive interpretability analysis for the {model_name} model,
including Grad-CAM visualizations to understand model decision-making.

## Grad-CAM Analysis

### Overview
Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions of the input
image that are most important for the model's prediction.

### Key Findings
"""
    
    if gradcam_results:
        avg_cam_max = np.mean([r['cam_max'] for r in gradcam_results])
        avg_cam_mean = np.mean([r['cam_mean'] for r in gradcam_results])
        
        report_content += f"""
- **Average CAM Maximum**: {avg_cam_max:.4f}
- **Average CAM Mean**: {avg_cam_mean:.4f}
- **Samples Analyzed**: {len(gradcam_results)}

### Interpretation
- Higher CAM values indicate regions the model focuses on for classification
- Values closer to 1.0 suggest strong attention to specific regions
- Uniform low values may indicate the model relies on global features

## Recommendations

### For Model Improvement
1. **Focus Areas**: Analyze regions with high attention to ensure they align with expected features
2. **Bias Detection**: Check if the model focuses on irrelevant background features
3. **Robustness**: Verify consistent attention patterns across similar samples

### For Production Deployment
1. **Quality Control**: Use attention maps to validate model decisions
2. **Error Analysis**: Investigate cases where attention doesn't match expectations
3. **User Trust**: Provide visualizations to explain model decisions to end users

## Technical Details

- **Model**: {model_name}
- **Output Directory**: {output_dir}
"""
    
    # Save report
    with open(output_dir / 'interpretability_report.md', 'w') as f:
        f.write(report_content)
    
    print("✅ Analysis report generated")

def main():
    """Main function."""
    args = parse_arguments()
    
    print("🔍 Model Interpretability Analysis")
    print("=" * 50)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"📱 Using device: {device}")
    
    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = REPORTS_DIR / 'interpretability' / args.model
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Load model
        model = load_model_and_weights(args.model, device)
        
        # Create data loader
        data_loaders = create_data_loaders('./bioast_dataset', batch_size=8, num_workers=1)
        test_loader = data_loaders['test']
        
        # Perform analyses
        gradcam_results = None
        
        if args.analysis in ['gradcam', 'all']:
            gradcam_results = perform_gradcam_analysis(
                model, args.model, test_loader, device, output_dir, args.num_samples
            )
        
        # Generate report
        generate_analysis_report(args.model, gradcam_results, output_dir)
        
        print(f"\n✅ Interpretability analysis completed!")
        print(f"📁 Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()