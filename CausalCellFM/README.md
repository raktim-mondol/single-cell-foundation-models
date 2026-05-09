# CausalCellFM: Counterfactual Perturbation Foundation Model

## Overview
CausalCellFM is designed for causal inference in single-cell perturbation data, using DE-weighted loss against perturb-seq hold-outs to learn causal relationships rather than mere correlations.

## Gaps Addressed
- **A**: Models are correlational, not causal
- **F**: No independent experimental validation

## Modular Structure

### Core Components

#### `model.py` - Model Architecture
- **CausalCellFM**: Encoder-decoder architecture with causal reasoning
- **PerturbationEncoder**: Encodes perturbation specifications
- **CounterfactualDecoder**: Generates counterfactual predictions
- **InvariantRepresentation**: Learns batch-invariant representations

#### `train.py` - Training Pipeline
- **PerturbSeqDataset**: Dataset wrapper for perturb-seq data
- **train_epoch()**: Training loop with multi-objective loss
- **eval_epoch()**: Evaluation with perturbation metrics
- **causal_direction_test()**: Tests causal direction consistency

#### `utils.py` - Utilities
- **set_seed()**: Reproducibility setup
- **generate_synthetic_perturbseq()**: Generates synthetic perturbation data
- **compute_perturbation_metrics()**: Computes evaluation metrics

## Usage

### Synthetic Data Test
```bash
python train.py --n_genes 1000 --n_cells 3000 --epochs 20
```

### Real Data Integration
```bash
python train.py --h5ad your_perturb_data.h5ad --n_genes 1000 --epochs 20
```

### Key Parameters
- `--n_genes`: Number of genes (default: 1000)
- `--n_cells`: Number of synthetic cells (default: 3000)
- `--n_batches`: Number of batches/donors (default: 5)
- `--d_model`: Model dimension (default: 128)
- `--enc_layers`: Encoder layers (default: 3)
- `--dec_layers`: Decoder layers (default: 3)
- `--epochs`: Training epochs (default: 20)

## Synthetic Data Generation
Generates synthetic perturb-seq dataset with:
- **Control expression**: Negative binomial distribution
- **Perturbation specification**: Random gene targets with directions (KO, WT, OE)
- **Perturbed expression**: Simulated knock-out/overexpression effects
- **DE mask**: Differentially expressed gene markers
- **Batch IDs**: Random batch assignments

## Data Format Expected
- `ctrl_expr`: [N, n_genes] - Control expression
- `pert_expr`: [N, n_genes] - Post-perturbation expression
- `pert_genes`: [N, K] - Gene indices of perturbation targets
- `pert_dirs`: [N, K] - Direction (0=KO, 1=WT, 2=OE)
- `pert_mags`: [N, K] - Magnitude of perturbation
- `de_mask`: [N, n_genes] - Differentially expressed genes
- `batch_ids`: [N] - Batch/donor labels

## Evaluation Metrics
- **Pearson Correlation (DE)**: Correlation on differentially expressed genes
- **Pearson Correlation (All)**: Correlation on all genes
- **MSE (DE)**: Mean squared error on DE genes
- **MSE (All)**: Mean squared error on all genes
- **Causal Direction Test**: Consistency of OE vs KO effects

## Model Architecture
1. **Input**: Control expression + perturbation specification
2. **Encoding**: Perturbation encoder + control encoder
3. **Counterfactual Generation**: Decoder predicts perturbed expression
4. **Loss**: DE-weighted MSE + invariance loss + KL divergence
5. **Evaluation**: Causal direction consistency checks

## Key Features
- Causal reasoning through counterfactual prediction
- DE-weighted loss for focused learning on relevant genes
- Batch invariance for robust cross-donor performance
- Causal direction testing for validation

## Dependencies
- torch
- numpy
- scipy