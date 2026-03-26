# CMS Detector Super-Resolution (GSoC 2026 - ML4SCI)

This repository contains my solution for the **Specific Task 2b** of the CMS projects.

## Project Goal:
The goal is to perform super-resolution on 3-channel calorimeter jet images, transforming **64x64 (Low Resolution)** inputs into **125x125 (High Resolution)** target images using Deep Learning.

## Initial Results:
I trained a CNN-based Generator for 100 batches as a proof of concept. 
- Loss (MSE) started at: 0.063
- Final Loss (after 100 batches): 0.021

### Visualization:
![Success Plot](basari_grafigi.png)

## How to Run:
1. Install dependencies: `pip install torch pandas pyarrow matplotlib`
2. Run `training.py` to train the model.
3. Run `test_results.py` to visualize results.
