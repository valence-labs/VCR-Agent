# Notebooks for Rubric Testing

This directory contains notebooks designed to test and evaluate the evaluation system located in `src/explain/eval/`. 

The primary goal of these notebooks is to validate that our evaluation rubrics work correctly and provide meaningful assessments. This includes:

-   **Rubric Functionality**: Testing individual rubrics (`CorrectnessRubric`, `FalsificationRubric`, `PlausibilityRubric`, `ToolUsageRubric`) to ensure they produce expected scores and feedback.
-   **LangSmith Integration**: Using the `LangSmithRubricWrapper` to run evaluations on LangSmith for tracking and analysis.
-   **Performance Validation**: Comparing rubric outputs against known ground truth to validate evaluation quality.

## Contents
-   `test-rubrics.ipynb`: Main notebook demonstrating rubric testing, dataset creation, and LangSmith integration
-   Helper scripts and utilities for evaluation workflows