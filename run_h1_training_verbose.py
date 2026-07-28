#!/usr/bin/env python3
"""
Run H1 training with output logging
"""
import subprocess
import sys
import os
import time

os.chdir('D:\\Bot_Trading\\production')

print("=" * 60)
print("Starting H1 Model Training Script")
print("=" * 60)
print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Working Directory: {os.getcwd()}")
print("=" * 60)

# Run the training script with unbuffered output
process = subprocess.Popen(
    [sys.executable, '-u', 'train_h1_model.py',
     '--data-path', 'D:\\Bot_Trading\\data\\okx_1h.csv',
     '--train-end', '2025-01-01',
     '--test-start', '2025-01-01'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

stdout_lines = []
stderr_lines = []

# Read stdout
try:
    while True:
        line = process.stdout.readline()
        if not line:
            break
        print(line, end='', flush=True)
        stdout_lines.append(line)
except:
    pass

# Wait and get remaining stderr
try:
    stderr, _ = process.communicate(timeout=30)
    if stderr:
        print("STDERR OUTPUT:")
        print(stderr)
        stderr_lines.append(stderr)
except:
    process.kill()

returncode = process.returncode

print("\n" + "=" * 60)
print(f"Training completed with exit code: {returncode}")
print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

sys.exit(returncode)
