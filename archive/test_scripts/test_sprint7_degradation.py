"""
Test the performance degradation detection system
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
    create_degradation_analyzer, DegradationType, SeverityLevel,
    DetectionMethod, PerformanceBaseline, DetectionConfig, PerformanceProfiler
)


class TestModel(nn.Module):
    """Simple test model"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test_performance_degradation():
    """Test performance degradation detection functionality"""
    print("Testing Performance Degradation Detection System")
    print("=" * 50)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Create degradation analyzer
        analyzer = create_degradation_analyzer(db_path=str(temp_dir / "degradation.db"))
        
        # Create test model
        model = TestModel()
        model.eval()
        
        # Establish performance baseline
        print("\nEstablishing performance baseline...")
        baseline_data = {
            "accuracy": [0.85, 0.86, 0.84, 0.87, 0.85, 0.86, 0.85, 0.84, 0.86, 0.85] * 20,
            "latency": [50, 52, 48, 51, 49, 50, 51, 49, 50, 52] * 20,
            "error_rate": [0.15, 0.14, 0.16, 0.13, 0.15, 0.14, 0.15, 0.16, 0.14, 0.15] * 20
        }
        
        baselines = analyzer.establish_baseline(
            model_id="test_model",
            version_id="v1.0",
            metric_data=baseline_data,
            min_samples=50
        )
        
        print(f"Established baselines for {len(baselines)} metrics")
        for metric_name, baseline in baselines.items():
            print(f"  {metric_name}: mean={baseline.mean:.4f}, std={baseline.std:.4f}")
        
        # Add custom detection configurations
        print("\nAdding detection configurations...")
        custom_configs = [
            DetectionConfig(
                metric_name="accuracy",
                degradation_type=DegradationType.ACCURACY_DROP,
                detection_method=DetectionMethod.STATISTICAL,
                threshold=0.05,
                min_samples=20,
                window_size=20,
                statistical_test="t_test",
                sensitivity=2.0
            ),
            DetectionConfig(
                metric_name="latency",
                degradation_type=DegradationType.LATENCY_INCREASE,
                detection_method=DetectionMethod.THRESHOLD_BASED,
                threshold=0.2,
                min_samples=20,
                window_size=20,
                statistical_test="z_score",
                sensitivity=2.0
            ),
            DetectionConfig(
                metric_name="memory_usage",
                degradation_type=DegradationType.MEMORY_LEAK,
                detection_method=DetectionMethod.TREND_ANALYSIS,
                threshold=0.1,
                min_samples=50,
                window_size=50,
                statistical_test="trend",
                sensitivity=1.5
            )
        ]
        
        for config in custom_configs:
            analyzer.add_detection_config(config)
        
        # Simulate performance recording and degradation detection
        print("\nSimulating performance monitoring...")
        np.random.seed(42)  # For reproducible results
        
        # Phase 1: Normal performance
        print("\nPhase 1: Normal performance")
        for i in range(30):
            # Normal performance around baseline
            accuracy = np.random.normal(0.85, 0.02)
            latency = np.random.normal(50, 3)
            error_rate = np.random.normal(0.15, 0.02)
            
            analyzer.record_performance("test_model", "v1.0", "accuracy", accuracy)
            analyzer.record_performance("test_model", "v1.0", "latency", latency)
            analyzer.record_performance("test_model", "v1.0", "error_rate", error_rate)
            
            if (i + 1) % 10 == 0:
                print(f"  Recorded {i + 1} normal performance samples")
        
        # Check for degradation (should be none)
        events = analyzer.detect_degradation("test_model", "v1.0")
        print(f"  Detected {len(events)} degradation events")
        
        # Phase 2: Gradual accuracy degradation
        print("\nPhase 2: Gradual accuracy degradation")
        for i in range(30):
            # Gradually decreasing accuracy
            base_acc = 0.85 - (i * 0.005)  # 0.85 to 0.70
            accuracy = np.random.normal(base_acc, 0.02)
            latency = np.random.normal(50, 3)
            error_rate = np.random.normal(0.15, 0.02)
            
            analyzer.record_performance("test_model", "v1.0", "accuracy", accuracy)
            analyzer.record_performance("test_model", "v1.0", "latency", latency)
            analyzer.record_performance("test_model", "v1.0", "error_rate", error_rate)
            
            if (i + 1) % 10 == 0:
                print(f"  Current accuracy: {accuracy:.4f}")
        
        # Check for degradation
        events = analyzer.detect_degradation("test_model", "v1.0")
        print(f"  Detected {len(events)} degradation events")
        
        # Show degradation events
        for event in events:
            print(f"\n  Degradation Event:")
            print(f"    Type: {event.degradation_type.value}")
            print(f"    Metric: {event.metric_name}")
            print(f"    Severity: {event.severity.value}")
            print(f"    Current: {event.current_value:.4f}")
            print(f"    Baseline: {event.baseline_value:.4f}")
            print(f"    Score: {event.degradation_score:.2%}")
            print(f"    Description: {event.description}")
            
            if event.root_causes:
                print("    Root Causes:")
                for cause in event.root_causes:
                    print(f"      - {cause}")
            
            if event.recommendations:
                print("    Recommendations:")
                for rec in event.recommendations:
                    print(f"      - {rec}")
        
        # Phase 3: Memory leak simulation
        print("\nPhase 3: Memory leak simulation")
        memory_base = 100
        for i in range(60):
            # Gradually increasing memory usage
            memory = memory_base + (i * 2) + np.random.normal(0, 5)
            analyzer.record_performance("test_model", "v1.0", "memory_usage", memory)
            
            if (i + 1) % 20 == 0:
                print(f"  Memory usage: {memory:.2f}MB")
        
        # Check for degradation
        events = analyzer.detect_degradation("test_model", "v1.0")
        memory_events = [e for e in events if e.degradation_type == DegradationType.MEMORY_LEAK]
        print(f"  Detected {len(memory_events)} memory leak events")
        
        # Performance profiling
        print("\nPerformance profiling...")
        profiler = PerformanceProfiler(model)
        
        # Create test input
        test_input = torch.randn(8, 3, 32, 32)
        
        # Profile inference
        profile_results = profiler.profile_inference(
            input_data=test_input,
            warmup_runs=5,
            profile_runs=20
        )
        
        print(f"  Average latency: {profile_results['avg_latency_ms']:.2f}ms")
        print(f"  P95 latency: {profile_results['p95_latency_ms']:.2f}ms")
        print(f"  Throughput: {profile_results['throughput_qps']:.2f} QPS")
        print(f"  Peak memory: {profile_results['peak_memory_mb']:.2f}MB")
        
        # Get active degradations
        print("\nActive degradation events:")
        active_events = analyzer.get_active_degradations("test_model")
        for event in active_events:
            print(f"  - {event.metric_name}: {event.severity.value} ({event.degradation_score:.2%})")
        
        # Generate degradation report
        print("\nGenerating degradation report...")
        report_path = analyzer.generate_degradation_report(
            model_id="test_model",
            output_path=str(temp_dir / "degradation_report.md")
        )
        print(f"Report saved to: {report_path}")
        
        # Show report summary
        with open(report_path, 'r') as f:
            report_content = f.read()
            print("\nReport Summary:")
            print("-" * 30)
            # Show first part of report
            summary_end = report_content.find("## Active Degradation Events")
            if summary_end != -1:
                print(report_content[:summary_end])
            else:
                print(report_content[:1000] + "...")
        
        # Test event resolution
        if active_events:
            print("\nTesting event resolution...")
            event_to_resolve = active_events[0]
            analyzer.resolve_degradation(
                event_to_resolve.id,
                "Model retrained with fresh data"
            )
            print(f"  Resolved event: {event_to_resolve.id}")
        
        print("\n✓ Performance degradation detection test completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_performance_degradation()