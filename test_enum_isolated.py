#!/usr/bin/env python3
"""Test the enum from the actual file"""

import sys
import os
import tempfile
import shutil

# Create a minimal version without imports
temp_content = '''
from enum import Enum

class RollbackTrigger(Enum):
    """回滚触发条件"""
    DEGRADATION_DETECTED = "degradation_detected"  # 检测到性能降级
    ERROR_RATE_SPIKE = "error_rate_spike"  # 错误率激增
    LATENCY_THRESHOLD = "latency_threshold"  # 延迟超过阈值
    MANUAL_TRIGGER = "manual_trigger"  # 手动触发
    HEALTH_CHECK_FAILED = "health_check_failed"  # 健康检查失败

# Test the enum
if __name__ == "__main__":
    print("Testing RollbackTrigger enum:")
    for trigger in RollbackTrigger:
        print(f"  {trigger.name} = {trigger.value}")
    
    print("\\nTesting DEGRADATION_DETECTED:")
    try:
        print(RollbackTrigger.DEGRADETION_DETECTED)
    except Exception as e:
        print(f"Error: {e}")
'''

# Write to temp file
with open('/tmp/test_enum.py', 'w') as f:
    f.write(temp_content)

# Run it
os.system('python3 /tmp/test_enum.py')