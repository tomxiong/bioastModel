#!/usr/bin/env python3
"""Check line 718"""

import sys
sys.path.insert(0, '/home/aaa/ws/bioastModel')

# Read the file and show line 718
with open('/home/aaa/ws/bioastModel/fua/production/auto_rollback.py', 'r') as f:
    lines = f.readlines()
    
print(f"Line 718: {repr(lines[717])}")

# Check for any non-printable characters
print("\nLine 718 character by character:")
for i, c in enumerate(lines[717]):
    if c == ' ':
        print(f"  {i}: ' ' (space)")
    else:
        print(f"  {i}: '{c}' (ord: {ord(c)})")