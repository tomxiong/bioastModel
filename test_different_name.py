#!/usr/bin/env python3
"""Test with different enum name"""

from enum import Enum

class TestTrigger(Enum):
    DEGRADATION_DETECTED = "degradation_detected"

print("Testing with different class name:")
try:
    print(TestTrigger.DEGRADETION_DETECTED)
except Exception as e:
    print(f"Error: {e}")

# Try accessing with the typo
print("\nTrying with typo:")
try:
    print(TestTrigger.DEGRADETION_DETECTED)
except Exception as e:
    print(f"Expected error: {e}")