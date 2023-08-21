# Explainable Representation Learning of Quantum States


## Project overview
This repository contains code to replicate the results in [Frohnert et al. (2023)](https://arxiv.org/abs/2306.05694).
In this work we investigate the potential of a generative machine learning model to develop interpretable representations of a quantum system. 
Through our research, we found that the model effectively learns to capture the underlying entanglement characteristics of quantum data, demonstrating its ability to (re)discover the entanglement measure concurrence.
This work highlights the ability of unsupervised machine learning models to produce interpretable representations of non-trivial features in quantum systems.



## Files in the repository:
        Quantum-State-VAE
          |-- data
                |-- *.h5 # trained VAE models 
                |-- *.npy # saved quantum states  
          |-- Analysis.ipynb # contains code to reproduce figures from paper
          |-- Minimal Example.ipynb # minimal example of VAE finding structure in quantum states
          |-- vae.py # module for VAE   
          |-- README
