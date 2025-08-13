# Asset Pricing via Transformer Latent Factor Models

## Overview
This repository contains the implementation of my Master's thesis, which extends the classical latent‐factor framework for empirical asset pricing.  
The main novelty is the integration of temporal firm characteristics and Transformer-based attention mechanisms into conditional latent factor models (TLFM).

We develop two Transformer-based variants:

1. **ChronosBolt-TLFM**  
   - Applies a pretrained ChronosBolt encoder to each reduced characteristic channel independently.  
   - Pools outputs via **cross-attention** to extract predictive signals.

2. **iTransformer-TLFM**  
   - Treats the full $L$-period history of each characteristic as a **variate token**.  
   - Learns temporal dependencies directly through inter-series attention.

Both models construct tradable factors and apply a **subsampled factor regularization** scheme to ensure $K$ in-sample uncorrelated factors, preserving economic interpretability while enhancing predictive power.

---

## Main Results (1980–2020 US Equities)
- Best performance achieved with only **K = 3 factors**, indicating a compact yet powerful factor structure.e.
- Factor regularization significantly boosts performance of both Conditional Autoencoders (CA) and our TLFM variants.

---

## Repository Structure

```
benchmark/
│ CA.py # Conditional Autoencoder baseline
│ IPCA.py # Instrumented PCA baseline

chronosbolt-TLFM/
│ Beta.py # Factor construction & beta estimation
│ CB_CA.py # ChronosBolt-based Conditional Autoencoder
│ data.py # Data loader for ChronosBolt model
│
└── chronos-forecasting/ # ChronosBolt forecasting utilities

data/
│ data_sorting.py # Sorting and grouping firm characteristics
│ features.json # Feature metadata
│ list_of_features.pdf # List & description of characteristics
│ maindata.py # Main data preparation
│ managed-portfolios.py # Portfolio construction
│ preprocessing.py # Data preprocessing utilities

iTransformer-TLFM/
│ data.py # Data loader for iTransformer model
│ iBeta.py # Factor construction for iTransformer
│ iCA.py # iTransformer-based Conditional Autoencoder
```
