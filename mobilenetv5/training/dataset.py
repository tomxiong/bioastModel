"""
MobileNetV5 Dataset Loader
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
from pathlib import Path
from typing import Optional, Tuple


class ColonyDataset(Dataset):
    """Dataset for colony detection images"""
    
    def __init__(self, data_dir: str, split: str = 'train', transform: Optional[transforms.Compose] = None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load image paths and labels
        self.images = []
        self.labels = []
        
        # Positive class (with colonies)
        positive_dir = self.data_dir / 'positive' / split
        if positive_dir.exists():
            for img_path in positive_dir.glob('*.png'):
                self.images.append(str(img_path))
                self.labels.append(1)  # Positive class
        
        # Negative class (no colonies/with pores)
        negative_dir = self.data_dir / 'negative' / split
        if negative_dir.exists():
            for img_path in negative_dir.glob('*.png'):
                self.images.append(str(img_path))
                self.labels.append(0)  # Negative class
        
        print(f"Loaded {len(self.images)} images for {split} split")
        print(f"Positive samples: {sum(self.labels)}")
        print(f"Negative samples: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label


class BrightnessNormalizationTransform:
    """Custom transform for brightness normalization to handle varying lighting conditions"""
    
    def __init__(self, target_mean: float = 0.5, target_std: float = 0.2):
        self.target_mean = target_mean
        self.target_std = target_std
    
    def __call__(self, tensor):
        # tensor is [C, H, W] in range [0, 1]
        if tensor.dim() != 3:
            return tensor
        
        # Calculate current statistics
        current_mean = tensor.mean()
        current_std = tensor.std()
        
        # Avoid division by zero
        if current_std < 1e-6:
            current_std = 1e-6
        
        # Normalize to target statistics
        normalized = (tensor - current_mean) / current_std
        normalized = normalized * self.target_std + self.target_mean
        
        # Clip to valid range
        normalized = torch.clamp(normalized, 0.0, 1.0)
        
        return normalized


class BiomedicalAugmentation:
    """Specialized augmentation for biomedical colony detection images"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, img):
        # Keep as PIL Image for transforms.ToTensor() to work properly
        # Apply random augmentations with probability p
        
        if random.random() < self.p:
            # Simulate varying bacterial solution transparency
            brightness_factor = 0.8 + 0.4 * random.random()
            img = transforms.functional.adjust_brightness(img, brightness_factor)
        
        if random.random() < self.p:
            # Simulate different lighting conditions
            contrast_factor = 0.8 + 0.4 * random.random()
            img = transforms.functional.adjust_contrast(img, contrast_factor)
        
        if random.random() < self.p:
            # Add slight blur to simulate different focus conditions
            img = transforms.functional.gaussian_blur(img, kernel_size=3, sigma=(0.1, 0.5))
        
        return img


def get_transforms(split: str = 'train', image_size: int = 70, use_enhanced_preprocessing: bool = True):
    """Get enhanced data transforms for training/validation"""
    
    if use_enhanced_preprocessing:
        if split == 'train':
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),  # Reduced rotation for medical images
                BiomedicalAugmentation(p=0.3),
                transforms.ColorJitter(
                    brightness=0.3,  # Increased brightness variation
                    contrast=0.3,   # Increased contrast variation  
                    saturation=0.1,  # Reduced saturation variation (medical images)
                    hue=0.05         # Minimal hue variation
                ),
                transforms.ToTensor(),
                BrightnessNormalizationTransform(target_mean=0.5, target_std=0.2),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                BrightnessNormalizationTransform(target_mean=0.5, target_std=0.2),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    else:
        # Original transforms for comparison
        if split == 'train':
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


def create_dataloaders(data_dir: str, batch_size: int = 32, image_size: int = 70, 
                      num_workers: int = 4, use_enhanced_preprocessing: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders with enhanced preprocessing"""
    
    # Create datasets with enhanced preprocessing
    train_dataset = ColonyDataset(data_dir, split='train', 
                                transform=get_transforms('train', image_size, use_enhanced_preprocessing))
    val_dataset = ColonyDataset(data_dir, split='val', 
                              transform=get_transforms('val', image_size, use_enhanced_preprocessing))
    test_dataset = ColonyDataset(data_dir, split='test', 
                               transform=get_transforms('val', image_size, use_enhanced_preprocessing))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    print(f"Created dataloaders with enhanced preprocessing: {use_enhanced_preprocessing}")
    return train_loader, val_loader, test_loader


def test_dataset():
    """Test dataset functionality"""
    data_dir = "../bioast_dataset"  # Adjust path as needed
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(data_dir, batch_size=4)
        
        # Test train loader
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels: {labels}")
            break
            
        print("Dataset test successful!")
        
    except Exception as e:
        print(f"Dataset test failed: {e}")


if __name__ == "__main__":
    test_dataset()