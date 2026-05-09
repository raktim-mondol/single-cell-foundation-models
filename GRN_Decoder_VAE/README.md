# GRN-Decoder VAE: GRN-Constrained Generative Foundation Model

## Overview
GRN-Decoder VAE uses gene regulatory network (GRN) constraints in the decoder computation graph, ensuring that the generative model respects known biological regulatory relationships.

## Gaps Addressed
- **A**: Models are correlational, not causal
- **B**: Biological priors are discarded

## Modular Structure

### Core Components

#### `model.py` - Model Architecture
- **GRNDecoderVAE**: VAE with GRN-constrained decoder
- **GRNEncoder**: Encodes expression to latent space
- **GRNDecoder**: Decoder with GRN computation graph
- **GRNLayer**: Implements GRN constraints in decoder
- **ZINBHead**: Zero-inflated negative binomial distribution head

#### `train.py` - Training Pipeline
- **CountDataset**: Dataset wrapper for count matrices
- **train_epoch()**: Training loop with ELBO optimization
- **val_epoch()**: Validation loop
- **edge_recovery_precision_recall()**: Evaluates GRN edge recovery
- **in_silico_knockout()**: In silico TF knockout simulation

#### `utils.py` - Utilities
- **set_seed()**: Reproducibility setup
- **generate_synthetic_grn_data()**: Generates synthetic GRN and expression data
- **evaluate_calibration()**: Evaluates prediction interval calibration
- **compute_edge_recovery()**: Computes GRN edge recovery metrics

## Usage

### Synthetic Data Test
```bash
python train.py --n_genes 200 --n_cells 2000 --grn_density 0.03 --epochs 30
```

### Real Data Integration
```bash
python train.py --h5ad your_data.h5ad --grn_matrix your_grn.csv --epochs 30
```

### Key Parameters
- `--n_genes`: Number of genes (default: 200)
- `--n_cells`: Number of cells (default: 2000)
- `--d_model`: Model dimension (default: 64)
- `--d_latent`: Latent dimension (default: 16)
- `--grn_density`: Density of synthetic GRN (default: 0.03)
- `--l1_lambda`: L1 regularization strength (default: 0.01)
- `--n_enc_layers`: Number of encoder layers (default: 2)
- `--kl_warmup`: KL weight warmup epochs (default: 10)
- `--epochs`: Training epochs (default: 30)

## Synthetic Data Generation
Generates synthetic GRN data with:
- **GRN Adjacency Matrix**: Random sparse regulatory network
- **Expression Data**: Generated from GRN-informed process
- **TF-Target Relationships**: Directed regulatory edges
- **Regulatory Strength**: Variable edge weights
- **Noise**: Biological and technical noise

## Data Format Expected
- `counts`: [N, n_genes] - Gene expression count matrix
- `grn_adj`: [n_genes, n_genes] - GRN adjacency matrix
- `tf_list`: List of transcription factor indices

## Model Architecture

### Encoder
- **Input**: Raw count matrix
- **Processing**: Log1p transformation + normalization
- **Architecture**: Multi-layer transformer encoder
- **Output**: Latent representation (μ, σ)

### Decoder with GRN Constraints
- **Input**: Latent sample z
- **GRN Integration**: Decoder computation graph follows GRN structure
- **Regulatory Flow**: TF → Target gene connections
- **Output**: Reconstructed expression parameters

### Loss Functions
- **Reconstruction Loss**: ZINB negative log-likelihood
- **KL Divergence**: Latent space regularization
- **L1 Regularization**: Sparsity in learned GRN
- **GRN Consistency**: Penalty for violating known GRN edges

## Evaluation Metrics

### Edge Recovery
- **Precision**: Fraction of predicted edges that are true
- **Recall**: Fraction of true edges that are predicted
- **F1 Score**: Harmonic mean of precision and recall
- **AUPRC**: Area under precision-recall curve

### In Silico Validation
- **KO Consistency**: TF knockout effects match predictions
- **Directionality**: Up/down regulation correctness
- **Magnitude**: Effect size accuracy

### Calibration
- **Coverage**: Prediction interval coverage probability
- **ECE**: Expected calibration error
- **Reliability Diagram**: Visual calibration assessment

## Key Features
- GRN-constrained generation for biological validity
- Interpretable latent space
- Causal reasoning through GRN structure
- Calibrated uncertainty quantification
- In silico perturbation capabilities

## Dependencies
- torch
- numpy
- scipy