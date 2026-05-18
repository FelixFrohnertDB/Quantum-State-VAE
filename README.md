# Explainable Representation Learning of Quantum States

[![MLST](https://img.shields.io/badge/MLST-10.1088/2632--2153/ad16a0-blue.svg)](https://doi.org/10.1088/2632-2153/ad16a0)


This repository contains the codebase to replicate the results presented in **[Frohnert et al. (2023)](https://arxiv.org/abs/2306.05694)**.

The project investigates the capacity of unsupervised, generative machine learning models (VAEs) to discover interpretable representations of quantum systems. Crucially, the model autonomously learns to capture underlying entanglement characteristics, effectively (re)discovering the quantum entanglement measure known as **concurrence**.

---

## Setup & Installation

Install the project and its dependencies in editable mode:

```bash
pip install -e .

```

---

## Repository Structure

```text
Quantum-State-VAE/
├── data/
│   ├── *.h5                # Pre-trained VAE models
│   └── *.npy               # Saved quantum states data
├── src/
│   └── vae_utils.py        # Core VAE architecture and helper modules
├── Analysis.ipynb          # Notebook to reproduce all figures from the paper
├── Minimal Example.ipynb   # A streamlined demo of the VAE finding structure in quantum states
└── README.md

```