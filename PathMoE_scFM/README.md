# PathMoE-scFM: Pathway-Aware Sparse Mixture-of-Experts Transformer

## Overview
PathMoE-scFM replaces monolithic feed-forward layers in transformers with sparse MoE layers where each expert corresponds to a curated biological pathway (Reactome, Hallmark, KEGG, TF-regulon from DoRothEA).

## Gaps Addressed
- **B**: Biological priors are discarded
- **D**: Poorly calibrated uncertainty (expert-disagreement signal)

## Modular Structure

### Core Components

#### `model.py` - Model Architecture
- **GeneTokeniser**: Maps count matrices to gene token sequences
- **PathwayExpert**: Single pathway expert (2-layer MLP)
- **PathwayMoE**: Sparse mixture-of-experts with pathway-based routing
- **PathMoETransformerBlock**: Transformer block with PathwayMoE
- **PathMoEscFM**: Full model with masked gene prediction objective
- **CellTypeClassifier**: Fine-tuning head for cell-type annotation

#### `train.py` - Training Pipeline
- **SingleCellDataset**: Dataset wrapper for AnnData count matrices
- **AnnotatedSingleCellDataset**: Supervised dataset for fine-tuning
- **pretrain()**: Pretraining loop with masked gene prediction
- **finetune()**: Fine-tuning loop for cell-type classification
- **evaluate_pathway_recovery()**: Evaluates pathway membership recovery

#### `utils.py` - Utilities
- **set_seed()**: Reproducibility setup
- **build_dummy_pathway_matrix()**: Creates random gene-pathway membership matrix
- **load_anndata()**: Loads real AnnData datasets
- **compute_pathway_recovery()**: Evaluates expert routing quality

## Usage

### Synthetic Data Test
```bash
python train.py --n_genes 2000 --n_cells 5000 --pretrain_epochs 20 --finetune_epochs 10
```

### Real Data Integration
```bash
python train.py --h5ad your_data.h5ad --n_genes 2000 --pretrain_epochs 20 --finetune_epochs 10
```

### Key Parameters
- `--n_genes`: Number of genes in vocabulary
- `--n_cells`: Number of synthetic cells (ignored if --h5ad provided)
- `--n_experts`: Number of pathway experts (default: 50)
- `--top_k`: Active experts per token (default: 4)
- `--d_model`: Model dimension (default: 128)
- `--pretrain_epochs`: Pretraining epochs (default: 20)
- `--finetune_epochs`: Fine-tuning epochs (default: 10)

## Synthetic Data Generation
The method automatically generates synthetic count data using negative binomial distribution when no `--h5ad` file is provided:
- Control expression: Negative binomial (n=2, p=0.5)
- Cell-type labels: Random integer assignments (0-9)
- Pathway matrix: Random binary membership (~3 pathways per gene)

## Evaluation Metrics
- **Pathway Recovery**: Precision@1 and Recall@5 for expert routing
- **Cell-type Accuracy**: Classification accuracy during fine-tuning
- **Load Balance**: Expert utilization uniformity

## Model Architecture
1. **Input**: Raw count matrix [batch, n_genes]
2. **Tokenisation**: Discretise expression → gene tokens
3. **Encoding**: Transformer blocks with PathwayMoE layers
4. **Pretraining**: Masked gene expression prediction
5. **Fine-tuning**: Cell-type classification via CLS token

## Key Features
- Biological prior integration via pathway-based expert initialization
- Sparse activation for computational efficiency
- Load balancing loss for uniform expert utilization
- Transfer learning via CLS token embeddings

## Dependencies
- torch
- numpy
- (Optional) anndata, scanpy for real data loading