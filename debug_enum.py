#!/usr/bin/env python3
"""Debug script for RollbackTrigger enum"""

import sys
sys.path.insert(0, '/home/aaa/ws/bioastModel')

# Import directly
from fua.production.auto_rollback import RollbackTrigger

print("RollbackTrigger enum values:")
for trigger in RollbackTrigger:
    print(f"  {trigger.name} = {trigger.value}")

print("\nTesting access:")
try:
    print(f"DEGRADATION_DETECTED exists: {RollbackTrigger.DEGRADETION_DETECTED}")
except AttributeError as e:
    print(f"Error: {e}")

# Check if we have the typo version
print("\nChecking for typo version:")
try:
    print(f"DEGRADTION_DETECTED (typo) exists: {RollbackTrigger.DEGRADETION_DETECTED}")
except AttributeError as e:
    print(f"Expected error for typo: {e}")