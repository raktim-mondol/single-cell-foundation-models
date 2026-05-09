# Atlas-Streamer: Continual Learning for Atlas Updates

## Overview
Atlas-Streamer enables continual learning as new single-cell atlas data releases become available, using self-distillation and importance-weighted replay to maintain performance on old cell types while learning new ones.

## Gaps Addressed
- **E**: Static models in a streaming-data world

## Modular Structure

### Core Components

#### `model.py` - Model Architecture
- **AtlasStreamer**: Continual learning framework with experience replay
- **SimpleScFMBackbone**: Base single-cell foundation model
- **SelfDistillationModule**: Knowledge distillation from teacher to student
- **ImportanceWeightedBuffer**: Experience replay with importance sampling
- **DynamicVocabulary**: Handles growing gene vocabulary over time

#### `train.py` - Training Pipeline
- **generate_streaming_releases()**: Simulates atlas data releases
- **update()**: Continual learning update for new data
- **evaluate_cell_type_accuracy()**: Evaluates backward/forward transfer
- **simulate_continual_learning()**: Main continual learning loop

#### `utils.py` - Utilities
- **set_seed()**: Reproducibility setup
- **generate_streaming_releases()**: Generates sequential atlas releases
- **evaluate_cell_type_accuracy()**: Cell-type classification accuracy
- **compute_forgetting()**: Measures catastrophic forgetting

## Usage

### Synthetic Data Test
```bash
python train.py --n_releases 5 --cells_per_release 500 --update_steps 100
```

### Real Data Integration
```bash
python train.py --data_dir path/to/atlas/releases --n_releases 5
```

### Key Parameters
- `--n_genes_init`: Initial gene vocabulary size (default: 500)
- `--gene_growth`: New genes added per release (default: 50)
- `--n_releases`: Number of simulated releases (default: 5)
- `--cells_per_release`: Cells per release (default: 500)
- `--n_cell_types_init`: Initial cell types (default: 8)
- `--new_types_per_release`: New cell types per release (default: 2)
- `--buffer_cap`: Experience replay buffer capacity (default: 5000)
- `--update_steps`: Training steps per release (default: 100)

## Synthetic Data Generation
Simulates streaming atlas releases with:
- **Growing vocabulary**: New genes added over time
- **New cell types**: Emergence of novel cell types
- **Donor drift**: Changing donor distributions
- **Batch effects**: Technical variations between releases
- **Concept drift**: Evolving biological understanding

## Continual Learning Strategy

### Self-Distillation
- Teacher model: Previous release's model
- Student model: Current release's model
- Knowledge transfer via distillation loss
- Preserves knowledge from previous releases

### Importance-Weighted Replay
- Experience replay buffer with diverse samples
- Importance sampling based on:
  - Cell-type rarity
  - Prediction uncertainty
  - Gradient magnitude
- Balances new and old knowledge

### Dynamic Vocabulary
- Handles growing gene vocabularies
- Embedding expansion for new genes
- Vocabulary alignment across releases
- Backward compatibility for old genes

## Evaluation Metrics
- **Backward Transfer**: Performance on original cell types
- **Forward Transfer**: Zero-shot performance on new types
- **Forgetting Measure**: Performance drop on old tasks
- **Learning Efficiency**: Steps needed per release
- **Memory Efficiency**: Buffer utilization

## Model Architecture
1. **Initial Training**: Pretrain on first atlas release
2. **Release Arrival**: New data with expanded vocabulary
3. **Vocabulary Update**: Expand embeddings for new genes
4. **Experience Replay**: Sample from buffer with importance weights
5. **Self-Distillation**: Teacher guides student with old knowledge
6. **Joint Optimization**: New data loss + distillation loss + replay loss
7. **Model Update**: Replace teacher with new student

## Key Features
- Continual learning without catastrophic forgetting
- Handles growing vocabularies and new cell types
- Efficient memory usage via importance-weighted replay
- Self-distillation for knowledge preservation
- Scalable to large-scale atlas updates

## Dependencies
- torch
- numpy