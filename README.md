# Data Drift Analyzer 🔍

Fast, robust, and visualization-rich data drift detection using Polars! 🚀

Built with AI 💙 via Claude Sonnet 3.5 for data scientists who love clean, fast, and reliable analytics!

## Key Features

- Core Analysis 📊
  - Like-to-like comparisons (matched on keys)
  - Full dataset drift analysis
  - Smart auto-detection of numeric & categorical features
  - Blazing fast with Polars backend

- Statistical Tests 📈
  - KS-test for numeric distributions
  - Chi-square for categorical features
  - P-value significance reporting
  - Distribution difference quantification

- Smart Visualizations 🎨
  - Auto-adapting plots based on data characteristics:
    - KDE plots for normal distributions
    - Histograms for non-normal data
    - Bar plots for categorical features
    - Special handling for constant values
  - Summary statistics overlay
  - Top-10 category comparisons

- Edge Cases & Error Handling 🛡️
  - Zero variance handling
  - Missing category detection
  - Null value management
  - Graceful error recovery

- Performance First 🏃‍♂️
  - Smart memory management
  - Parallel processing ready

- Rich Reporting 📝
  - Comprehensive drift summaries
  - Statistical significance highlights
  - Category-specific changes
  - Sample size analysis

## Quick Start 🚀

```python
analyzer = DataDriftAnalyzer(
    train_df=training_data,
    inf_df=inference_data,
    join_keys=['correlation_id', 'hotel_id']
)

# Run analysis
joined_results, overall_results = analyzer.analyze_drift()

# Get summary
analyzer.print_drift_summary(joined_results, "Joined Analysis")
analyzer.print_drift_summary(overall_results, "Overall Analysis")
```

## Perfect For 🎯
- ML Model Monitoring
- Data Quality Checks
- Feature Drift Detection
- Distribution Analysis
- Production Data Validation

## Requirements ⚙️
- polars
- numpy
- scipy
- matplotlib
- seaborn
