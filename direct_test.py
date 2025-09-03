#!/usr/bin/env python3
"""Direct import test"""

import sys
import os
import importlib.util

# Load the module directly
spec = importlib.util.spec_from_file_location("auto_rollback", "/home/aaa/ws/bioastModel/fua/production/auto_rollback.py")
auto_rollback = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auto_rollback)

# Test the enum
print("Testing direct import:")
RollbackTrigger = auto_rollback.RollbackTrigger

try:
    trigger = RollbackTrigger.DEGRADETION_DETECTED
    print(f"✓ Success: {trigger}")
except Exception as e:
    print(f"✗ Error: {e}")

# Check enum definition
print("\nEnum definition:")
print(RollbackTrigger.__dict__)