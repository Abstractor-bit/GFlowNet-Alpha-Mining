# GFlowNet for Alpha Factor Mining

**Author**: He Hongjin (何泓锦)  
**Affiliation**: HKUST AI Major | Prospective Stanford Exchange Summer 2026  
**Date**: March 2026

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

This project applies **Generative Flow Networks (GFlowNets)** to discover diverse, high-quality formulaic alpha factors in quantitative finance. Unlike traditional RL methods (PPO, DDPG) that converge to single local optima, GFlowNets enable **proportional sampling** from multiple high-reward modes, generating diverse alpha portfolios robust to market regime changes.

## 🔬 Motivation

**Problem**: Traditional optimization methods for alpha discovery suffer from:
- Mode collapse (converge to single best factor)
- Overfitting to historical patterns
- Lack of diversity (high correlation between discovered factors)

**Solution**: GFlowNet samples diverse alphas proportional to their Information Coefficient (IC), naturally balancing exploration and exploitation.

**Key Advantage**: Robust factor portfolio with low inter-factor correlation → better generalization and higher Sharpe ratios.

## 🏗️ Implementation

### Environment
- **State**: Partial alpha expression (sequence of features/operators)
- **Action space**: 8 market features + 4 operators {+, -, *, /} + STOP token
- **Termination**: Maximum depth reached or STOP action
- **Reward**: Absolute Information Coefficient (correlation with forward returns)

### Model Architecture
- **Forward policy**: 3-layer MLP (130-dim input → 128 hidden → 13 actions)
- **Loss function**: Trajectory Balance (TB) loss
- **Optimizer**: Adam (lr=1e-3)
- **Training**: 500 episodes with epsilon-greedy exploration

### Features
8 market features across multiple timescales:
- Returns: 1-day, 5-day, 20-day
- Volatility: 5-day, 20-day rolling std
- Volume: ratio to 20-day moving average
- Price: close, volume (raw)

## 📊 Results

**Baseline Implementation (Week 1):**
- Mean IC: [YOUR_VALUE] (vs random baseline: ~0.05)
- Max IC: [YOUR_VALUE]
- Diversity: [X] unique expressions from 50 samples
- Training time: <5 minutes on Google Colab

**Top Generated Alpha Example:**


## 🔍 Critical Analysis & Limitations

### Identified Issues
1. **No train/test split**: Current IC likely optimistically biased (20-40% expected decay OOS)
2. **Simple state encoding**: Sequence representation loses expression tree structure
3. **Single objective**: IC only, ignoring transaction costs, capacity, risk
4. **Small dataset**: 5 stocks, 4 years (vs production: 100s of stocks, 10+ years)
5. **Unsafe evaluation**: Using `eval()` instead of proper AST parser

### Planned Improvements
- [ ] Implement time-series cross-validation (priority #1)
- [ ] RGCN encoder for structure-aware representation (AlphaSAGE approach)
- [ ] Multi-objective reward: IC + Sharpe + novelty - turnover
- [ ] Integrate local credit assignment (Pan et al. 2024)
- [ ] Distributional GFlowNet for risk-aware generation

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch 2.0+
- **Data**: yfinance (prototype), target Qlib (production)
- **Numerical**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn

## 🚀 Quick Start

### Run in Google Colab (Recommended)
```bash
# Open the notebook
https://colab.research.google.com/github/YOUR_USERNAME/GFlowNet-Alpha-Mining/blob/main/notebooks/Alpha_GFN_v1_Baseline.ipynb

# Run all cells
Runtime → Run all

git clone https://github.com/YOUR_USERNAME/GFlowNet-Alpha-Mining.git
cd GFlowNet-Alpha-Mining
pip install -r requirements.txt
jupyter notebook notebooks/Alpha_GFN_v1_Baseline.ipynb
