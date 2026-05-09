# scTrueBench: Causal Benchmark Suite

## Overview
scTrueBench provides a comprehensive benchmark suite for evaluating single-cell foundation models across causal reasoning, calibration, and wet-lab validation axes.

## Gaps Addressed
- **All**: Addresses all six gaps through comprehensive evaluation

## Modular Structure

### Core Components

#### `benchmark.py` - Benchmark Framework
- **BenchmarkData**: Data structure for benchmark datasets
- **ScTrueBench**: Main benchmark evaluation framework
- **CausalEvaluator**: Causal reasoning evaluation
- **CalibrationEvaluator**: Uncertainty calibration evaluation
- **WetLabRegistry**: Wet-lab experiment tracking

#### `run_benchmark.py` - Benchmark Runner
- **random_model_fn()**: Baseline random embeddings
- **pca_model_fn()**: Baseline PCA embeddings
- **main()**: Benchmark execution pipeline
- Model integration examples

#### `utils.py` - Utilities
- **set_seed()**: Reproducibility setup
- **generate_benchmark_dataset()**: Generates comprehensive benchmark data
- **compute_causal_metrics()**: Causal evaluation metrics
- **compute_calibration_metrics()**: Calibration evaluation metrics

## Usage

### Quick Benchmark Test
```bash
python run_benchmark.py --n_cells 500 --n_genes 200
```

### Full Benchmark Evaluation
```bash
python run_benchmark.py --n_cells 2000 --n_genes 1000 --n_types 10 --n_batches 5
```

### Custom Model Integration
```python
# Define your model function
def my_model_fn(counts: np.ndarray) -> np.ndarray:
    # Your model implementation
    return embeddings

# Run benchmark
bench.run(
    model_name="MyModel",
    embeddings=embeddings,
    pred_pert_expr=perturbations,
    pred_samples=mc_samples,
    model_fn=my_model_fn,
    oe_delta=oe_predictions,
    ko_delta=ko_predictions
)
```

### Key Parameters
- `--n_cells`: Number of cells (default: 500)
- `--n_genes`: Number of genes (default: 200)
- `--n_types`: Number of cell types (default: 5)
- `--n_batches`: Number of batches (default: 3)
- `--registry_path`: Path for wet-lab registry (default: "wetlab_registry_demo")

## Benchmark Axes

### 1. Causal Reasoning
- **Perturbation Prediction Accuracy**: OE/KO effect prediction
- **Direction Consistency**: Up/down regulation correctness
- **Magnitude Accuracy**: Effect size prediction
- **Counterfactual Validity**: Causal relationship learning

### 2. Calibration
- **Coverage Probability**: Prediction interval coverage
- **Expected Calibration Error (ECE)**: Overall calibration
- **Reliability Diagram**: Visual calibration assessment
- **Sharpness**: Prediction interval width

### 3. Wet-Lab Validation
- **Prediction Registry**: Track model predictions
- **Outcome Tracking**: Record experimental results
- **Leaderboard**: Model ranking by wet-lab performance
- **Reproducibility**: Standardized experiment tracking

## Data Format Expected

### Benchmark Data
- `counts_ctrl`: [N, n_genes] - Control expression
- `counts_pert`: [N, n_genes] - Perturbed expression
- `pert_genes`: [N, K] - Perturbed gene indices
- `pert_dirs`: [N, K] - Perturbation directions
- `de_mask`: [N, n_genes] - DE gene mask
- `cell_type_labels`: [N] - Cell type labels
- `batch_labels`: [N] - Batch labels
- `grn_adj`: [n_genes, n_genes] - GRN adjacency (optional)

### Model Outputs Required
- `embeddings`: [N, d_model] - Cell embeddings
- `pred_pert_expr`: [N, n_genes] - Predicted perturbed expression
- `pred_samples`: [N_samples, N, n_genes] - Monte Carlo samples
- `oe_delta`: [n_genes] - Overexpression delta predictions
- `ko_delta`: [n_genes] - Knockout delta predictions

## Evaluation Metrics

### Causal Metrics
- **Pearson Correlation**: Predicted vs actual perturbations
- **Spearman Correlation**: Rank correlation
- **Mean Absolute Error**: Effect size accuracy
- **Direction Accuracy**: Up/down prediction correctness
- **AUPRC**: Area under precision-recall curve

### Calibration Metrics
- **Coverage**: Prediction interval coverage at various confidence levels
- **ECE**: Expected calibration error
- **Brier Score**: Probabilistic prediction accuracy
- **Reliability**: Calibration curve analysis

### Wet-Lab Metrics
- **Confirmation Rate**: Fraction of predictions confirmed
- **Precision**: Confirmed predictions / total predictions
- **Recall**: Confirmed predictions / true effects
- **F1 Score**: Harmonic mean of precision and recall

## Model Integration

### Step 1: Define Model Function
```python
def model_fn(counts: np.ndarray) -> np.ndarray:
    """
    Args:
        counts: Gene expression matrix [N, n_genes]
    Returns:
        embeddings: Cell embeddings [N, d_model]
    """
    # Your implementation
    pass
```

### Step 2: Generate Predictions
```python
# Get embeddings
embeddings = model_fn(benchmark_data.counts_ctrl)

# Predict perturbations (if applicable)
pred_pert_expr = your_perturbation_model(...)

# Generate Monte Carlo samples (for calibration)
pred_samples = your_model.sample_predictions(...)
```

### Step 3: Run Benchmark
```python
result = bench.run(
    model_name="YourModel",
    embeddings=embeddings,
    pred_pert_expr=pred_pert_expr,
    pred_samples=pred_samples,
    model_fn=model_fn,
    oe_delta=oe_predictions,
    ko_delta=ko_predictions
)
```

### Step 4: Register Wet-Lab Predictions
```python
pred = WetLabPrediction(
    model_name="YourModel",
    prediction_id=unique_id,
    gene_targets=[0, 1, 2],
    perturbation="KO",
    predicted_top_up=[...],
    predicted_top_down=[...],
    cell_line="K562"
)
registry.register_prediction(pred)
```

## Wet-Lab Integration

### Prediction Registration
- Register predictions before wet-lab experiments
- Include model metadata and experimental details
- Track prediction confidence and rationale

### Outcome Submission
- Submit experimental results
- Link to original prediction ID
- Include experimental conditions

### Leaderboard
- Automatic ranking by wet-lab performance
- Compare models across experiments
- Track reproducibility and consistency

## Key Features
- Comprehensive evaluation across multiple axes
- Standardized benchmark protocols
- Wet-lab validation integration
- Model-agnostic evaluation framework
- Reproducible benchmarking pipeline

## Dependencies
- torch
- numpy
- scipy
- sklearn