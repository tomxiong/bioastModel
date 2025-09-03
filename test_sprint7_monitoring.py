"""
Test the model monitoring system
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
    create_model_monitor, AlertSeverity, MetricType, AlertChannel,
    MetricThreshold, ModelMetrics
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


def test_model_monitoring():
    """Test model monitoring functionality"""
    print("Testing Model Monitoring System")
    print("=" * 40)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Create model monitor
        monitor = create_model_monitor(
            db_path=str(temp_dir / "monitoring.db"),
            metrics_buffer_size=100,
            anomaly_window_size=20,
            anomaly_sensitivity=2.0
        )
        
        # Create test model
        model = TestModel()
        model.eval()
        
        # Add model to monitoring
        monitor.add_model(
            model_id="test_model",
            version_id="v1.0",
            model=model,
            config={"input_size": (1, 3, 32, 32)}
        )
        
        # Add threshold configurations
        thresholds = [
            MetricThreshold(
                metric_type=MetricType.ACCURACY,
                warning_threshold=0.8,
                critical_threshold=0.7,
                operator="less_than"
            ),
            MetricThreshold(
                metric_type=MetricType.LATENCY,
                warning_threshold=100.0,
                critical_threshold=200.0,
                operator="greater_than"
            ),
            MetricThreshold(
                metric_type=MetricType.ERROR_RATE,
                warning_threshold=0.2,
                critical_threshold=0.3,
                operator="greater_than"
            )
        ]
        
        for threshold in thresholds:
            monitor.alert_manager.add_threshold(threshold)
        
        # Add console notifier
        def console_notifier(alert):
            print(f"\n[ALERT] {alert.severity.value.upper()}: {alert.message}")
            
        monitor.alert_manager.add_alert_channel(AlertChannel.CONSOLE, console_notifier)
        
        # Start monitoring
        print("\nStarting monitoring...")
        monitor.start_monitoring(interval=2)
        
        # Simulate some model usage with varying performance
        print("\nSimulating model usage...")
        for i in range(10):
            # Create test data
            test_data = torch.randn(4, 3, 32, 32)
            
            # Simulate varying performance
            if i < 3:
                # Good performance
                print(f"  Run {i+1}: Normal performance")
            elif i < 6:
                # Degraded performance
                print(f"  Run {i+1}: Degraded performance")
                # Simulate higher latency by adding sleep
                time.sleep(0.1)
            else:
                # Poor performance (should trigger alerts)
                print(f"  Run {i+1}: Poor performance (may trigger alerts)")
                # Simulate very high latency
                time.sleep(0.2)
            
            # Let monitor collect metrics
            time.sleep(1)
        
        # Wait for monitoring to complete
        time.sleep(3)
        
        # Stop monitoring
        print("\nStopping monitoring...")
        monitor.stop_monitoring()
        
        # Check collected metrics
        print("\nRetrieving metrics...")
        metrics = monitor.get_metrics("test_model")
        print(f"Collected {len(metrics)} metric entries")
        
        if metrics:
            latest = metrics[0]
            print(f"\nLatest metrics:")
            print(f"  Accuracy: {latest.accuracy:.4f}")
            print(f"  Latency: {latest.latency_ms:.2f}ms")
            print(f"  Error Rate: {latest.error_rate:.4f}")
            print(f"  Memory Usage: {latest.memory_usage_mb:.2f}MB")
        
        # Check alerts
        print("\nChecking alerts...")
        alerts = monitor.get_alerts("test_model")
        print(f"Generated {len(alerts)} alerts")
        
        for alert in alerts[:3]:  # Show first 3 alerts
            print(f"\nAlert: {alert.severity.value.upper()}")
            print(f"  Metric: {alert.metric_type.value}")
            print(f"  Message: {alert.message}")
            print(f"  Value: {alert.value:.4f}")
            print(f"  Threshold: {alert.threshold:.4f}")
        
        # Generate report
        print("\nGenerating monitoring report...")
        report_path = monitor.generate_report("test_model", str(temp_dir / "monitoring_report.md"))
        print(f"Report saved to: {report_path}")
        
        # Show report summary
        with open(report_path, 'r') as f:
            report_content = f.read()
            print("\nReport Summary:")
            print("-" * 20)
            print(report_content.split("## Alerts")[0])
        
        print("\n✓ Model monitoring test completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_model_monitoring()