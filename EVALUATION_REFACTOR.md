# Evaluation Refactor Summary

## Overview
Consolidated the baseline and RLHF evaluation workflows into a unified evaluation system with enhanced features for comprehensive model assessment.

## Changes Made

### 1. Removed Baseline Workflow
- **Deleted**: `searchlm/workflows/baseline/` directory
  - `baseline.py` - Baseline query generation
  - `__init__.py` - Package initialization
- **Reason**: The RLHF evaluation code was simpler and more flexible, capable of evaluating both base and fine-tuned models

### 2. Enhanced RLHF Evaluation (`searchlm/workflows/rlhf/evaluation.py`)

#### New Features:
1. **Unified Model Evaluation**
   - Single codebase evaluates both base models (e.g., `Qwen/Qwen2.5-3B-Instruct`) and RLHF checkpoints
   - Configurable model paths for flexible evaluation

2. **JSON Audit Logs**
   - All evaluation results saved as JSON to volume mounts
   - Files saved to `modal_data/outputs/evaluations/{base|rlhf}/`
   - Each run includes timestamp, model info, and complete metrics

3. **Multiple Evaluation Runs**
   - Run evaluations multiple times to assess variability
   - Configurable number of runs per model
   - Default: 3 runs per model

4. **Aggregate Statistics**
   - Compute mean, std, min, max across multiple runs
   - Detailed per-metric statistics for each dataset
   - Side-by-side comparison of base vs RLHF models

5. **Comprehensive Reporting**
   - Aggregate metrics printed for each model
   - Direct comparison showing improvement percentages
   - Summary JSON with all runs and aggregate statistics

#### New Functions:

```python
def save_evaluation_results(
    results: dict,
    model_name: str,
    model_path: str,
    run_number: int,
    output_dir: Path,
) -> Path:
    """Save evaluation results as JSON to volume mount."""

def compute_aggregate_metrics(all_runs: list[dict]) -> dict:
    """Compute aggregate statistics across multiple evaluation runs."""

def print_aggregate_results(aggregate: dict, model_name: str):
    """Print aggregate statistics in a readable format."""

def evaluate_single_run(
    model_path: str,
    model_name: str,
    run_number: int,
    output_dir: Path,
) -> dict:
    """Evaluate a model on all configured datasets (single run)."""

def evaluate_multiple_runs(
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    checkpoint_path: Optional[str] = None,
    num_runs: int = 3,
    evaluate_base: bool = True,
    evaluate_rlhf: bool = True,
) -> dict:
    """Run comprehensive evaluation with multiple runs for both base and RLHF models."""

def print_comparison(all_results: dict):
    """Print side-by-side comparison of base vs RLHF models."""
```

#### Backward Compatibility:
- Original `evaluate()` function retained for compatibility
- Updated to use new infrastructure internally
- Single-run evaluation still supported

### 3. Updated Documentation

#### README.md
- Removed baseline generation workflow section
- Updated workflows section to focus on unified evaluation
- Added examples of using `evaluate_multiple_runs()`
- Updated feature list to highlight new evaluation capabilities
- Updated project structure to remove baseline directory

#### MODAL_DEVELOPMENT.md
- Replaced all baseline workflow examples with evaluation examples
- Updated hot reload examples to show evaluation iteration
- Updated interactive shell examples
- Updated all test scenarios to use evaluation instead of baseline
- Added examples for comprehensive evaluation with multiple runs

### 4. Output Structure

```
modal_data/outputs/evaluations/
├── base/
│   ├── base_eval_run1_20260131_143022.json
│   ├── base_eval_run2_20260131_143045.json
│   └── base_eval_run3_20260131_143108.json
├── rlhf/
│   ├── rlhf_eval_run1_20260131_143130.json
│   ├── rlhf_eval_run2_20260131_143153.json
│   └── rlhf_eval_run3_20260131_143216.json
└── evaluation_summary_20260131_143216.json
```

Each JSON file contains:
```json
{
  "model_name": "base",
  "model_path": "Qwen/Qwen2.5-3B-Instruct",
  "run_number": 1,
  "timestamp": "20260131_143022",
  "results": {
    "nfcorpus": {
      "ndcg@10": 0.3456,
      "ndcg@100": 0.3912,
      "mrr": 0.4123,
      "map": 0.2134,
      "precision@10": 0.0823,
      "recall@10": 0.1245,
      "num_queries": 323,
      "num_failed": 2
    },
    "scifact": { ... }
  }
}
```

Summary file contains:
```json
{
  "base_model": "Qwen/Qwen2.5-3B-Instruct",
  "rlhf_checkpoint": "/path/to/checkpoint",
  "num_runs": 3,
  "timestamp": "20260131_143216",
  "results": {
    "base": {
      "runs": [ ... ],
      "aggregate": {
        "nfcorpus": {
          "ndcg@10": {
            "mean": 0.3456,
            "std": 0.0012,
            "min": 0.3441,
            "max": 0.3467,
            "values": [0.3456, 0.3441, 0.3467]
          },
          ...
        }
      }
    },
    "rlhf": { ... }
  }
}
```

## Usage Examples

### Basic Usage (Single Run)
```python
from searchlm.workflows.rlhf.evaluation import evaluate

# Evaluate latest RLHF checkpoint
results = evaluate()

# Evaluate specific checkpoint
results = evaluate(checkpoint_path="/path/to/checkpoint")
```

### Comprehensive Evaluation (Multiple Runs)
```python
from searchlm.workflows.rlhf.evaluation import evaluate_multiple_runs

# Default: 3 runs each for base and RLHF models
results = evaluate_multiple_runs()

# Custom number of runs
results = evaluate_multiple_runs(num_runs=5)

# Evaluate only base model
results = evaluate_multiple_runs(
    evaluate_base=True,
    evaluate_rlhf=False
)

# Evaluate only RLHF model
results = evaluate_multiple_runs(
    evaluate_base=False,
    evaluate_rlhf=True
)

# Custom base model
results = evaluate_multiple_runs(
    base_model_name="Qwen/Qwen2.5-7B-Instruct",
    checkpoint_path="/path/to/checkpoint",
    num_runs=5
)
```

### Command Line Usage
```bash
# Run evaluation
python -m searchlm.workflows.rlhf.evaluation

# Or via Modal
modal run modal_dev.py::run_evaluation
modal run modal_dev.py::run_comprehensive_evaluation
```

## Benefits

1. **Simplified Codebase**
   - Removed duplicate evaluation logic
   - Single source of truth for evaluation
   - Easier to maintain and extend

2. **Better Auditing**
   - All results saved as JSON for later analysis
   - Timestamped files for tracking evaluation history
   - Complete metadata for reproducibility

3. **Statistical Rigor**
   - Multiple runs capture evaluation variability
   - Aggregate statistics provide confidence intervals
   - Better understanding of model performance

4. **Flexible Comparison**
   - Direct base vs RLHF comparison
   - Improvement percentages calculated automatically
   - Can evaluate either or both models

5. **Developer Experience**
   - Simpler API with sensible defaults
   - Backward compatible with existing code
   - Works seamlessly with Modal hot reload

## Migration Guide

### Before (Baseline)
```python
from searchlm.workflows.baseline.baseline import BaselineGenerator

generator = BaselineGenerator(
    dataset_name="mteb/scifact",
    output_filepath="outputs/scifact_queries.tsv"
)
generator.generate()
```

### After (Unified Evaluation)
```python
from searchlm.workflows.rlhf.evaluation import evaluate_multiple_runs

# Evaluate base model (no RLHF needed)
results = evaluate_multiple_runs(
    base_model_name="Qwen/Qwen2.5-3B-Instruct",
    evaluate_base=True,
    evaluate_rlhf=False,
    num_runs=3
)

# Results automatically saved to modal_data/outputs/evaluations/
```

## Testing

To verify the changes work correctly:

```python
# Test single evaluation
from searchlm.workflows.rlhf.evaluation import evaluate
results = evaluate()
print(results)

# Test multiple runs
from searchlm.workflows.rlhf.evaluation import evaluate_multiple_runs
results = evaluate_multiple_runs(num_runs=2, evaluate_rlhf=False)
print(results["base"]["aggregate"])
```

## Future Enhancements

Potential improvements to consider:
1. Parallel execution of multiple runs
2. Configurable metrics and K values
3. Support for additional datasets
4. Visualization of aggregate statistics
5. Statistical significance testing between models
6. Export to other formats (CSV, Excel, etc.)

## Files Changed

### Deleted
- `searchlm/workflows/baseline/baseline.py`
- `searchlm/workflows/baseline/__init__.py`
- `searchlm/workflows/baseline/` (directory)

### Modified
- `searchlm/workflows/rlhf/evaluation.py` (major enhancement)
- `README.md` (documentation updates)
- `docs/MODAL_DEVELOPMENT.md` (documentation updates)

### Added
- This summary document (`EVALUATION_REFACTOR.md`)

## Conclusion

This refactor simplifies the codebase while adding powerful new evaluation capabilities. The unified approach makes it easier to compare base and fine-tuned models, provides better auditing through JSON logs, and offers statistical rigor through multiple runs and aggregate metrics.
