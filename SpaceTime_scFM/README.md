# SpaceTime-scFM: Spatio-Temporal Multi-Modal Foundation Model

## Overview
SpaceTime-scFM treats spatial coordinates and pseudotime as first-class tokens, enabling the model to learn from spatial transcriptomics data with temporal components for comprehensive cellular context understanding.

## Gaps Addressed
- **C**: Spatial and temporal context ignored
- **B**: Biological priors are discarded

## Modular Structure

### Core Components

#### `model.py` - Model Architecture
- **SpaceTimescFM**: Multi-modal transformer with spatial/temporal tokens
- **SpatialTokeniser**: Converts spatial coordinates to tokens
- **PseudotimeTokeniser**: Converts pseudotime to tokens
- **MultiModalFusion**: Fuses expression, spatial, and temporal information
- **NichePredictionHead**: Predicts cellular niche from context

#### `train.py` - Training Pipeline
- **SpatioTemporalDataset**: Dataset with variable modality presence
- **collate_fn()**: Handles missing spatial/temporal data
- **train_epoch()**: Training with mosaic objectives
- **val_epoch()**: Validation loop
- **evaluate_niche_prediction()**: Evaluates spatial niche prediction

#### `utils.py` - Utilities
- **set_seed()**: Reproducibility setup
- **generate_synthetic_spatial_data()**: Generates synthetic spatial transcriptomics data
- **evaluate_niche_prediction()**: Niche prediction accuracy metrics

## Usage

### Synthetic Data Test
```bash
python train.py --n_genes 500 --n_cells 2000 --spatial_dim 2 --epochs 20
```

### Real Data Integration
```bash
python train.py --h5ad your_spatial_data.h5ad --n_genes 500 --epochs 20
```

### Key Parameters
- `--n_genes`: Number of genes (default: 500)
- `--n_cells`: Number of cells (default: 2000)
- `--spatial_dim`: Spatial dimensions (2 for Visium, 3 for 3D MERFISH)
- `--spatial_frac`: Fraction of cells with spatial coordinates (default: 0.6)
- `--pt_frac`: Fraction of cells with pseudotime (default: 0.4)
- `--d_model`: Model dimension (default: 128)
- `--n_layers`: Number of transformer layers (default: 4)
- `--epochs`: Training epochs (default: 20)

## Synthetic Data Generation
Generates synthetic spatial transcriptomics data with:
- **Expression**: Negative binomial distribution
- **Spatial coordinates**: 2D/3D coordinates for subset of cells
- **Pseudotime**: Temporal ordering for subset of cells
- **Niche labels**: Spatial niche assignments
- **Variable modality**: Not all cells have spatial/temporal data

## Data Format Expected
- `expr`: [N, n_genes] - Gene expression matrix
- `coords`: [N, spatial_dim] - Spatial coordinates (optional)
- `pseudotime`: [N] - Pseudotime values (optional)
- `has_spatial`: [N] - Boolean mask for spatial data
- `has_pt`: [N] - Boolean mask for pseudotime data
- `niche_labels`: [N] - Spatial niche assignments

## Training Objectives
1. **Masked Gene Prediction**: Standard masked language modeling
2. **Masked Position Prediction**: Predict spatial coordinates when masked
3. **Pseudotime Regression**: Predict pseudotime from cellular context
4. **Niche Prediction**: Predict spatial niche from multi-modal features

## Evaluation Metrics
- **Reconstruction Loss**: Gene expression reconstruction
- **Spatial Prediction Accuracy**: Coordinate prediction accuracy
- **Pseudotime MSE**: Temporal prediction error
- **Niche Prediction Accuracy**: Spatial niche classification

## Model Architecture
1. **Input**: Expression + optional spatial coordinates + optional pseudotime
2. **Tokenisation**: Separate tokenisers for each modality
3. **Multi-modal Fusion**: Cross-attention between modalities
4. **Mosaic Training**: Variable modality presence per batch
5. **Multi-task Learning**: Joint optimization of all objectives

## Key Features
- First-class treatment of spatial and temporal information
- Handles missing modalities gracefully
- Mosaic training for robustness
- Multi-task learning for comprehensive understanding
- Cross-modal attention for information integration

## Dependencies
- torch
- numpy
- scipy