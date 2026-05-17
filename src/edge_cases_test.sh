#!/bin/bash

# Make the script exit immediately if any command fails
set -e

echo "--------------------------------------------------------"
echo "Starting 5 Edge-Case Validation Tests for train.py"
echo "--------------------------------------------------------"

# Test 1: Absolute Table Minimums
# Checking if the dimension math holds at the lowest possible channel/layer sizes
echo "Test 1: Table Minimums..."
python src/train.py --epochs 1 --batch_size 4 --conv_ch1 4 --conv_ch2 16 --conv_ch3 32 --fc1_dim 64 --fc2_dim 32 --run_name "test_min"

# Test 2: Absolute Table Maximums
# Checking for Memory (OOM) issues at highest capacity
echo "Test 2: Table Maximums..."
python src/train.py --epochs 1 --batch_size 512 --conv_ch1 32 --conv_ch2 128 --conv_ch3 256 --fc1_dim 512 --fc2_dim 256 --run_name "test_max"

# Test 3: Odd Numbers & Non-Powers of 2
# Ensuring DataLoader and Conv math handles non-standard sizes
echo "Test 3: Non-standard Integers..."
python src/train.py --epochs 1 --batch_size 7 --conv_ch1 13 --conv_ch2 41 --conv_ch3 83 --fc1_dim 127 --fc2_dim 85 --run_name "test_odd"

# Test 4: High Regularization (Floats)
# Checking if 0.5 Dropout + high Weight Decay causes vanishing gradients
echo "Test 4: High Regularization..."
python src/train.py --epochs 1 --batch_size 32 --dropout_rate 0.5 --weight_decay 0.01 --lr 0.1 --run_name "test_reg"

# Test 5: Minimum Duration (1 Epoch)
# Checking that the logging and final test evaluation work on a minimal run
echo "Test 5: Minimal Duration..."
python src/train.py --epochs 1 --batch_size 64 --run_name "test_brief"

echo "--------------------------------------------------------"
echo "All 5 tests passed successfully! Code is OMA-ready."
echo "--------------------------------------------------------"
