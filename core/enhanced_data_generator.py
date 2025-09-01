#!/usr/bin/env python3
"""
Enhanced Data Generator for 70x70 Biomedical Images
Creates more realistic synthetic biomedical data for MIC analysis
"""

import os
import numpy as np
import cv2
from typing import Tuple, List
import json
from datetime import datetime

class EnhancedBiomedicalDataGenerator:
    """Enhanced generator for realistic 70x70 biomedical images"""
    
    def __init__(self, output_dir: str = "data", image_size: Tuple[int, int] = (70, 70)):
        self.output_dir = output_dir
        self.image_size = image_size
        self.positive_dir = os.path.join(output_dir, 'positive')
        self.negative_dir = os.path.join(output_dir, 'negative')
        
        # Create directories
        os.makedirs(self.positive_dir, exist_ok=True)
        os.makedirs(self.negative_dir, exist_ok=True)
    
    def generate_realistic_biomedical_image(self, is_positive: bool = True) -> np.ndarray:
        """Generate a realistic 70x70 biomedical image"""
        
        # Create base image with realistic biomedical background
        image = np.ones((70, 70, 3), dtype=np.float32) * 0.85  # Light background
        
        # Add realistic well/plate structure
        center = (35, 35)
        well_radius = 32
        
        # Create circular well boundary
        y, x = np.ogrid[:70, :70]
        well_mask = (x - center[0])**2 + (y - center[1])**2 <= well_radius**2
        
        # Well interior - slightly darker
        image[well_mask] = 0.75
        
        # Add realistic lighting gradient
        for i in range(70):
            for j in range(70):
                dist_from_center = np.sqrt((i-35)**2 + (j-35)**2)
                if dist_from_center <= well_radius:
                    # Radial lighting effect
                    lighting_factor = 1.0 - (dist_from_center / well_radius) * 0.15
                    image[i, j] *= lighting_factor
        
        if is_positive:
            # Positive samples: Add bacterial growth patterns
            self._add_bacterial_growth(image, well_mask)
            
            # Add turbidity (cloudiness)
            turbidity_level = np.random.uniform(0.3, 0.7)
            self._add_turbidity(image, well_mask, turbidity_level)
            
            # Add some texture variation
            self._add_growth_texture(image, well_mask)
            
        else:
            # Negative samples: Clear medium
            # Add slight variations but keep mostly clear
            clear_variation = np.random.normal(0, 0.02, image.shape)
            image[well_mask] += clear_variation[well_mask]
            
            # Very light turbidity for negative samples
            light_turbidity = np.random.uniform(0.05, 0.15)
            self._add_turbidity(image, well_mask, light_turbidity)
        
        # Add realistic noise and artifacts
        self._add_imaging_artifacts(image)
        
        # Add edge effects around well
        self._add_well_edges(image, center, well_radius)
        
        # Ensure values are in valid range
        image = np.clip(image, 0, 1)
        
        return image
    
    def _add_bacterial_growth(self, image: np.ndarray, well_mask: np.ndarray):
        """Add bacterial growth patterns"""
        # Add multiple growth centers
        num_growth_centers = np.random.randint(2, 6)
        
        for _ in range(num_growth_centers):
            # Random growth center within well
            center_x = np.random.randint(15, 55)
            center_y = np.random.randint(15, 55)
            
            # Check if center is within well
            if (center_x - 35)**2 + (center_y - 35)**2 <= 30**2:
                growth_radius = np.random.randint(3, 12)
                growth_intensity = np.random.uniform(0.2, 0.5)
                
                # Create growth pattern
                y, x = np.ogrid[:70, :70]
                growth_mask = (x - center_x)**2 + (y - center_y)**2 <= growth_radius**2
                
                # Apply growth with gradient
                for i in range(70):
                    for j in range(70):
                        if growth_mask[i, j] and well_mask[i, j]:
                            dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                            if dist <= growth_radius:
                                fade_factor = 1.0 - (dist / growth_radius)
                                image[i, j] -= growth_intensity * fade_factor
    
    def _add_turbidity(self, image: np.ndarray, well_mask: np.ndarray, level: float):
        """Add turbidity (cloudiness) effect"""
        # Create turbidity pattern
        turbidity_noise = np.random.normal(0, level * 0.1, image.shape)
        
        # Apply only within well
        image[well_mask] -= level * 0.3
        image[well_mask] += turbidity_noise[well_mask]
    
    def _add_growth_texture(self, image: np.ndarray, well_mask: np.ndarray):
        """Add texture patterns typical of bacterial growth"""
        # Create streaky patterns
        for _ in range(np.random.randint(3, 8)):
            start_x = np.random.randint(10, 60)
            start_y = np.random.randint(10, 60)
            
            if (start_x - 35)**2 + (start_y - 35)**2 <= 25**2:
                # Create streak
                length = np.random.randint(5, 15)
                angle = np.random.uniform(0, 2 * np.pi)
                
                for step in range(length):
                    x = int(start_x + step * np.cos(angle))
                    y = int(start_y + step * np.sin(angle))
                    
                    if 0 <= x < 70 and 0 <= y < 70 and well_mask[y, x]:
                        intensity = np.random.uniform(0.1, 0.3)
                        image[y, x] -= intensity
    
    def _add_imaging_artifacts(self, image: np.ndarray):
        """Add realistic imaging artifacts"""
        # Add slight vignetting
        center = (35, 35)
        for i in range(70):
            for j in range(70):
                dist_from_center = np.sqrt((i - center[1])**2 + (j - center[0])**2)
                vignette_factor = 1.0 - (dist_from_center / 50) * 0.05
                image[i, j] *= max(vignette_factor, 0.95)
        
        # Add sensor noise
        noise = np.random.normal(0, 0.01, image.shape)
        image += noise
    
    def _add_well_edges(self, image: np.ndarray, center: Tuple[int, int], radius: int):
        """Add realistic well edge effects"""
        y, x = np.ogrid[:70, :70]
        
        # Create edge mask (ring around well)
        outer_mask = (x - center[0])**2 + (y - center[1])**2 <= (radius + 2)**2
        inner_mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        edge_mask = outer_mask & ~inner_mask
        
        # Darken edges slightly
        image[edge_mask] *= 0.9
    
    def generate_dataset(self, num_positive: int = 1000, num_negative: int = 1000):
        """Generate complete dataset with positive and negative samples"""
        
        print(f"🧬 Generating Enhanced Biomedical Dataset")
        print(f"📊 Positive samples: {num_positive}")
        print(f"📊 Negative samples: {num_negative}")
        print(f"📐 Image size: {self.image_size}")
        
        # Generate positive samples
        print("\n🦠 Generating positive samples (bacterial growth)...")
        for i in range(num_positive):
            image = self.generate_realistic_biomedical_image(is_positive=True)
            
            # Convert to 0-255 range for saving
            image_uint8 = (image * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
            
            # Save image
            filename = f"positive_{i:04d}.png"
            filepath = os.path.join(self.positive_dir, filename)
            cv2.imwrite(filepath, image_bgr)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_positive} positive samples")
        
        # Generate negative samples
        print("\n🧪 Generating negative samples (clear medium)...")
        for i in range(num_negative):
            image = self.generate_realistic_biomedical_image(is_positive=False)
            
            # Convert to 0-255 range for saving
            image_uint8 = (image * 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
            
            # Save image
            filename = f"negative_{i:04d}.png"
            filepath = os.path.join(self.negative_dir, filename)
            cv2.imwrite(filepath, image_bgr)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_negative} negative samples")
        
        # Create dataset info
        dataset_info = {
            "dataset_name": "Enhanced Biomedical MIC Dataset",
            "generated_at": datetime.now().isoformat(),
            "image_size": self.image_size,
            "total_samples": num_positive + num_negative,
            "positive_samples": num_positive,
            "negative_samples": num_negative,
            "positive_dir": self.positive_dir,
            "negative_dir": self.negative_dir,
            "description": "Realistic 70x70 biomedical images for MIC analysis with bacterial growth patterns"
        }
        
        # Save dataset info
        info_path = os.path.join(self.output_dir, 'dataset_info.json')
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n✅ Dataset generation completed!")
        print(f"📁 Positive samples: {self.positive_dir}")
        print(f"📁 Negative samples: {self.negative_dir}")
        print(f"📄 Dataset info: {info_path}")
        
        return dataset_info

def main():
    """Generate enhanced biomedical dataset"""
    generator = EnhancedBiomedicalDataGenerator()
    
    # Generate dataset with more samples for better training
    dataset_info = generator.generate_dataset(
        num_positive=1500,  # More positive samples
        num_negative=1500   # More negative samples
    )
    
    print(f"\n🎉 Enhanced biomedical dataset ready for training!")
    return dataset_info

if __name__ == "__main__":
    main()