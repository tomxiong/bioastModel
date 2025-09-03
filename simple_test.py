#!/usr/bin/env python3
"""Simple test for auto rollback"""

import sys
import os
sys.path.insert(0, '/home/aaa/ws/bioastModel')

from fua.production.auto_rollback import RollbackTrigger

# Test 1: Check if enum works
print("Testing RollbackTrigger enum:")
try:
    trigger = RollbackTrigger.DEGRADETION_DETECTED
    print(f"✓ DEGRADATION_DETECTED = {trigger}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Check all enum values
print("\nAll enum values:")
for t in RollbackTrigger:
    print(f"  {t.name} = {t.value}")

# Test 3: Try the typo
print("\nTesting typo:")
try:
    typo_trigger = RollbackTrigger.DEGRADETION_DETECTED
    print(f"Typo worked: {typo_trigger}")
except Exception as e:
    print(f"Expected error: {e}")