# Peak Time & Purchase Drop-off Analysis in eCommerce Events

## Objective
This Phase 1 project builds a cloud-based data pipeline on Azure to analyze eCommerce user behavior.

The project answers two questions:
1. What day of the week and hour of the day have the highest user activity?
2. At which stage does the largest drop-off occur in the sequential funnel: view -> cart -> purchase?

## Dataset
- Source: Kaggle - eCommerce Behavior Data from Multi-Category Store
- Type: Batch dataset
- Initial scope for Phase 1: one monthly file
- Key fields: event_time, event_type, user_id, user_session, product/category attributes

## Planned Azure Architecture
- Azure Storage Account
- Azure Data Factory
- Azure Databricks

## Repository Structure
- `docs/` project documentation
- `notebooks/` Databricks-aligned notebooks
- `src/` reusable Python utilities
- `outputs/` figures and summary tables
- `screenshots/` Azure evidence for submission

## Status
Project setup in progress.
