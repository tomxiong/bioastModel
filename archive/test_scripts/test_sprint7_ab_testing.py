"""
Test the A/B testing framework
"""

import torch
import torch.nn as nn
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fua.production import (
    create_ab_test_manager, ABTestConfig, TestVariant, TestMetric,
    TrafficAllocationStrategy, TestStatus, ABMetricType, StatisticalTest
)


class SimpleModel(nn.Module):
    """Simple test model"""
    def __init__(self, accuracy_rate=0.8):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 2)
        self.accuracy_rate = accuracy_rate
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test_ab_testing():
    """Test A/B testing functionality"""
    print("Testing A/B Testing Framework")
    print("=" * 40)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Create A/B test manager
        manager = create_ab_test_manager(db_path=str(temp_dir / "ab_tests.db"))
        
        # Create test configuration
        config = ABTestConfig(
            name="Model Comparison Test",
            description="Compare accuracy of two different models",
            traffic_allocation_strategy=TrafficAllocationStrategy.EQUAL,
            duration_days=7,
            significance_level=0.05,
            variants=[
                TestVariant(
                    id="control",
                    name="Control Model",
                    model_id="simple_model_v1",
                    version_id="v1.0",
                    weight=0.5,
                    is_control=True
                ),
                TestVariant(
                    id="treatment",
                    name="Treatment Model",
                    model_id="simple_model_v2",
                    version_id="v2.0",
                    weight=0.5
                )
            ],
            metrics=[
                TestMetric(
                    name="accuracy",
                    type=ABMetricType.ACCURACY,
                    primary=True,
                    improvement_direction="higher",
                    min_detectable_effect=0.05,
                    statistical_test=StatisticalTest.T_TEST
                ),
                TestMetric(
                    name="latency",
                    type=ABMetricType.LATENCY,
                    primary=False,
                    improvement_direction="lower",
                    min_detectable_effect=10.0,
                    statistical_test=StatisticalTest.T_TEST
                )
            ]
        )
        
        # Create test
        test = manager.create_test(config)
        print(f"\nCreated A/B test: {test.id}")
        print(f"Name: {config.name}")
        print(f"Variants: {len(config.variants)}")
        print(f"Metrics: {len(config.metrics)}")
        
        # Start test
        manager.start_test(test.id)
        print("\nTest started")
        
        # Simulate user traffic and metric collection
        print("\nSimulating user traffic...")
        np.random.seed(42)  # For reproducible results
        
        n_users = 200
        for i in range(n_users):
            user_id = f"user_{i}"
            
            # Allocate variant
            variant = manager.allocate_variant(test.id, user_id)
            
            # Simulate model prediction with different performance
            if variant.id == "control":
                # Control model: 80% accuracy
                accuracy = 0.8 + np.random.normal(0, 0.1)
                latency = 50 + np.random.normal(0, 10)
            else:
                # Treatment model: 85% accuracy (better)
                accuracy = 0.85 + np.random.normal(0, 0.1)
                latency = 45 + np.random.normal(0, 10)
            
            # Ensure values are in reasonable ranges
            accuracy = np.clip(accuracy, 0, 1)
            latency = max(1, latency)
            
            # Record metrics
            manager.record_metric(test.id, variant.id, "accuracy", accuracy, user_id)
            manager.record_metric(test.id, variant.id, "latency", latency, user_id)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{n_users} users")
        
        # Get test results
        print("\nAnalyzing results...")
        results = manager.get_test_results(test.id)
        
        # Display results
        for metric_name, metric_results in results.items():
            print(f"\n{metric_name.upper()} Results:")
            print("-" * 30)
            
            for result in metric_results:
                variant_name = "Control" if result.variant_id == "control" else "Treatment"
                print(f"\n{variant_name}:")
                print(f"  Value: {result.value:.4f}")
                print(f"  Count: {result.count}")
                print(f"  P-value: {result.p_value:.4f}")
                print(f"  Effect Size: {result.effect_size:.4f}")
                print(f"  Significant: {'Yes' if result.is_significant else 'No'}")
                print(f"  Winner: {'Yes' if result.is_winner else 'No'}")
        
        # Stop test
        manager.stop_test(test.id, "Test completed after collecting sufficient data")
        print("\nTest stopped")
        
        # Generate report
        report_path = manager.generate_report(test.id, str(temp_dir / "ab_test_report.md"))
        print(f"\nReport saved to: {report_path}")
        
        # Show report summary
        with open(report_path, 'r') as f:
            report_content = f.read()
            print("\nReport Summary:")
            print("-" * 20)
            # Show first part of report
            summary_end = report_content.find("## Results")
            if summary_end != -1:
                print(report_content[:summary_end])
            else:
                print(report_content[:1000] + "...")
        
        # Test sample size calculation
        print("\nSample Size Calculation:")
        print("-" * 30)
        sample_size = manager.calculate_sample_size(
            baseline_rate=0.8,
            min_detectable_effect=0.05,
            significance_level=0.05,
            power=0.8
        )
        print(f"Required sample size per variant: {sample_size}")
        
        # List all tests
        print("\nAll Tests:")
        print("-" * 30)
        tests = manager.list_tests()
        for test in tests:
            print(f"  {test.config.name} ({test.id}) - {test.status.value}")
        
        print("\n✓ A/B testing framework test completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_ab_testing()