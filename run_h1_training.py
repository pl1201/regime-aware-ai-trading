#!/usr/bin/env python3
"""
Run H1 training and stream output in real-time
"""
import subprocess
import sys
import os

os.chdir('D:\\Bot_Trading\\production')

# Run the training script
process = subprocess.Popen(
    [sys.executable, 'train_h1_model.py',
     '--data-path', 'D:\\Bot_Trading\\data\\okx_1h.csv',
     '--train-end', '2025-01-01',
     '--test-start', '2025-01-01'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Stream output line by line
for line in process.stdout:
    print(line, end='')

# Wait for completion
returncode = process.wait()
sys.exit(returncode)
