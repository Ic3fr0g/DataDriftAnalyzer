# DataDriftAnalyzer

Built with AI 💙 via Claude Sonnet 3.5 for Polars dataframes

## Quickstart

```python
# Initialize and run analysis
analyzer = DataDriftAnalyzer(train_data, inf_data, join_keys=["correlation_key"])
joined_results, overall_results = analyzer.analyze_drift()

# Print summaries
analyzer.print_drift_summary(joined_results, "Joined")
analyzer.print_drift_summary(overall_results, "Overall")
```
