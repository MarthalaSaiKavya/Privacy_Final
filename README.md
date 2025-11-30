# BOXRR-23 VR Motion + Differential Privacy Pipeline

This repository contains the full code for our CSC 533 project on **privacy-preserving VR motion analytics** using the **BOXRR-23** dataset.

We build an end-to-end pipeline that:

1. Loads a BOXRR-23 Beat Saber session (head + hand pose over time).
2. Performs exploratory data analysis (EDA) with plots and stats.
3. Converts frame-level pose data into **window-level motion segments**.
4. Labels each window as **high** vs **low motion**.
5. Trains **non-DP baselines** (Logistic Regression, MLP).
6. Adds **feature-level DP** (Laplace + Gaussian noise).
7. Trains **DP-SGD** models with Opacus.
8. Implements **PATE** (teacher ensemble + noisy vote aggregation).
9. Implements **PATE-GAN** (GAN backbone + PATE labels).
10. Implements **DP-GAN** (DP on the discriminator via Opacus).
11. Compares all mechanisms and exports a **summary CSV** of metrics.

