"""
Test the auto rollback system
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
    create_auto_rollback_manager, RollbackConfig, RollbackTrigger,
    create_degradation_analyzer, create_model_monitor,
    Alert, AlertSeverity, DegradationEvent, SeverityLevel,
    DegradationType
)


class TestModel(nn.Module):
    """Simple test model"""
    def __init__(self, version="v1.0"):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 2)
        self.version = version
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test_auto_rollback():
    """Test auto rollback functionality"""
    print("Testing Auto Rollback System")
    print("=" * 40)
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Create rollback configuration
        rollback_config = RollbackConfig(
            auto_rollback_enabled=True,
            degradation_threshold=0.05,  # 5%降级触发回滚
            error_rate_threshold=0.3,
            max_rollback_versions=3,
            canary_rollback_enabled=True
        )
        
        # Create auto rollback manager
        rollback_manager = create_auto_rollback_manager(
            storage_path=str(temp_dir / "model_versions"),
            config=rollback_config
        )
        
        # Create degradation analyzer and model monitor
        degradation_analyzer = create_degradation_analyzer(
            db_path=str(temp_dir / "degradation.db")
        )
        
        model_monitor = create_model_monitor(
            db_path=str(temp_dir / "monitoring.db")
        )
        
        # Set up event handlers
        def on_rollback_started(plan):
            print(f"\n[ROLLBACK] Starting rollback: {plan.id}")
            print(f"  From: {plan.from_version}")
            print(f"  To: {plan.to_version}")
            print(f"  Reason: {plan.reason}")
        
        def on_rollback_completed(plan):
            print(f"\n[ROLLBACK] Rollback completed: {plan.id}")
            print(f"  Strategy: {plan.rollback_strategy}")
            print(f"  Duration: {(plan.completed_at - plan.executed_at).total_seconds():.2f}s")
        
        def on_rollback_failed(plan):
            print(f"\n[ROLLBACK] Rollback failed: {plan.id}")
        
        rollback_manager.add_event_handler("rollback_started", on_rollback_started)
        rollback_manager.add_event_handler("rollback_completed", on_rollback_completed)
        rollback_manager.add_event_handler("rollback_failed", on_rollback_failed)
        
        # Save multiple model versions
        print("\nSaving model versions...")
        
        # Version 1.0 - Good performance
        model_v1 = TestModel("v1.0")
        version_v1 = rollback_manager.version_manager.save_version(
            model_id="test_model",
            version="v1.0",
            model=model_v1,
            metadata={"accuracy": 0.90, "error_rate": 0.10}
        )
        # Mark as stable
        rollback_manager.version_manager.mark_as_stable(version_v1.id, True)
        rollback_manager.version_manager.update_performance_metrics(
            version_v1.id, 
            {"accuracy": 0.90, "error_rate": 0.10, "latency": 50}
        )
        print(f"  Saved v1.0 (stable)")
        
        # Version 2.0 - Better performance
        model_v2 = TestModel("v2.0")
        version_v2 = rollback_manager.version_manager.save_version(
            model_id="test_model",
            version="v2.0",
            model=model_v2,
            metadata={"accuracy": 0.92, "error_rate": 0.08}
        )
        # Mark as stable
        rollback_manager.version_manager.mark_as_stable(version_v2.id, True)
        rollback_manager.version_manager.update_performance_metrics(
            version_v2.id,
            {"accuracy": 0.92, "error_rate": 0.08, "latency": 45}
        )
        print(f"  Saved v2.0 (stable)")
        
        # Version 3.0 - Will cause degradation
        model_v3 = TestModel("v3.0")
        version_v3 = rollback_manager.version_manager.save_version(
            model_id="test_model",
            version="v3.0",
            model=model_v3,
            metadata={"accuracy": 0.85, "error_rate": 0.15}
        )
        # Not marked as stable
        print(f"  Saved v3.0 (unstable)")
        
        # Set v3.0 as current
        rollback_manager.version_manager.set_current_version("test_model", version_v3.id)
        print(f"\nCurrent version: v3.0")
        
        # Test 1: Manual rollback
        print("\nTest 1: Manual rollback")
        print("-" * 30)
        
        success = rollback_manager.manual_rollback(
            model_id="test_model",
            strategy="immediate"
        )
        
        if success:
            current = rollback_manager.version_manager.get_current_version("test_model")
            print(f"✓ Manual rollback successful")
            print(f"  Current version: {current.version}")
        else:
            print("✗ Manual rollback failed")
        
        # Set v3.0 as current again for degradation test
        rollback_manager.version_manager.set_current_version("test_model", version_v3.id)
        
        # Test 2: Rollback triggered by degradation alert
        print("\nTest 2: Rollback triggered by degradation")
        print("-" * 40)
        
        # Create degradation event
        degradation_event = DegradationEvent(
            id="degradation_test_1",
            model_id="test_model",
            version_id=version_v3.id,
            degradation_type=DegradationType.ACCURACY_DROP,
            severity=SeverityLevel.HIGH,
            metric_name="accuracy",
            current_value=0.75,
            baseline_value=0.90,
            degradation_score=0.167,  # 16.7% drop
            detection_method=None,
            description="Accuracy dropped significantly"
        )
        
        # Handle degradation alert
        rollback_manager.handle_degradation_alert(degradation_event)
        
        # Wait for rollback to complete
        time.sleep(2)
        
        # Check current version
        current = rollback_manager.version_manager.get_current_version("test_model")
        print(f"Current version after degradation: {current.version}")
        
        # Test 3: Rollback triggered by monitoring alert
        print("\nTest 3: Rollback triggered by monitoring alert")
        print("-" * 45)
        
        # Create a problematic version
        model_v4 = TestModel("v4.0")
        version_v4 = rollback_manager.version_manager.save_version(
            model_id="test_model",
            version="v4.0",
            model=model_v4,
            metadata={"accuracy": 0.70, "error_rate": 0.30}
        )
        rollback_manager.version_manager.set_current_version("test_model", version_v4.id)
        
        # Create monitoring alert
        alert = Alert(
            id="alert_test_1",
            model_id="test_model",
            version_id=version_v4.id,
            metric_type=None,
            severity=AlertSeverity.CRITICAL,
            message="Error rate exceeded threshold: 35%",
            value=0.35,
            threshold=0.30
        )
        
        # Handle monitoring alert
        rollback_manager.handle_monitoring_alert(alert)
        
        # Wait for rollback to complete
        time.sleep(2)
        
        # Check current version
        current = rollback_manager.version_manager.get_current_version("test_model")
        print(f"Current version after alert: {current.version}")
        
        # Test 4: Canary rollback
        print("\nTest 4: Canary rollback")
        print("-" * 25)
        
        # Create another problematic version
        model_v5 = TestModel("v5.0")
        version_v5 = rollback_manager.version_manager.save_version(
            model_id="test_model",
            version="v5.0",
            model=model_v5,
            metadata={"accuracy": 0.72, "error_rate": 0.28}
        )
        rollback_manager.version_manager.set_current_version("test_model", version_v5.id)
        
        # Manual rollback with canary strategy
        success = rollback_manager.manual_rollback(
            model_id="test_model",
            strategy="canary"
        )
        
        if success:
            print("✓ Canary rollback successful")
        
        # Test 5: Version management
        print("\nTest 5: Version management")
        print("-" * 30)
        
        # List all versions
        print("\nAll versions:")
        for version_id, version in rollback_manager.version_manager.versions.items():
            current_mark = " (current)" if version.id == rollback_manager.version_manager.current_versions.get("test_model") else ""
            stable_mark = " ✓" if version.is_stable else ""
            print(f"  {version.version}: {version.health_score:.2f}{stable_mark}{current_mark}")
        
        # Get stable versions
        stable_versions = rollback_manager.version_manager.get_stable_versions("test_model")
        print(f"\nStable versions: {len(stable_versions)}")
        for v in stable_versions:
            print(f"  {v.version}: health={v.health_score:.2f}, rollbacks={v.rollback_count}")
        
        # Test 6: Rollback history
        print("\nTest 6: Rollback history")
        print("-" * 30)
        
        history = rollback_manager.get_rollback_history("test_model")
        print(f"Total rollbacks: {len(history)}")
        
        for plan in history[-3:]:  # Show last 3
            print(f"\n  {plan.trigger.value}: {plan.from_version[-8:]} -> {plan.to_version[-8:]}")
            print(f"    Status: {plan.status.value}")
            print(f"    Strategy: {plan.rollback_strategy}")
        
        # Generate rollback report
        print("\nGenerating rollback report...")
        report_path = rollback_manager.generate_rollback_report(
            output_path=str(temp_dir / "rollback_report.md")
        )
        print(f"Report saved to: {report_path}")
        
        # Show report summary
        with open(report_path, 'r') as f:
            report_content = f.read()
            print("\nReport Summary:")
            print("-" * 20)
            # Show first part of report
            summary_end = report_content.find("## Recent Rollbacks")
            if summary_end != -1:
                print(report_content[:summary_end])
            else:
                print(report_content[:1000] + "...")
        
        # Test 7: Load and test rolled back model
        print("\nTest 7: Load rolled back model")
        print("-" * 35)
        
        current_version = rollback_manager.version_manager.get_current_version("test_model")
        if current_version:
            # Load the model
            loaded_model = rollback_manager.version_manager.load_version(
                current_version.id,
                TestModel
            )
            print(f"✓ Successfully loaded model version: {current_version.version}")
            
            # Test inference
            test_input = torch.randn(1, 3, 32, 32)
            with torch.no_grad():
                output = loaded_model(test_input)
            print(f"✓ Model inference successful: output shape {output.shape}")
        
        print("\n✓ Auto rollback system test completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_auto_rollback()