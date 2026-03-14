# Phase 1 – eCommerce Behavior Data on Azure

## Project Overview
This project implements the Phase 1 data foundation for an analytics system using the **eCommerce Behavior Data from Multi-Category Store** dataset. The goal is to build a reproducible Azure-based pipeline that supports two project questions:

1. **Peak Time Analysis**: identify the day of the week and hour of the day with the highest number of user views.
2. **Purchase Drop-off Analysis**: identify where the largest drop-off occurs in the **sequential** customer funnel **View → Cart → Purchase**.

Each row in the dataset represents a user action such as `view`, `cart`, or `purchase`. The system uses event logs to extract time-based activity patterns and session-level funnel behavior.

---

## Project Objective
Phase 1 focuses on building the data foundation of the system. The implementation covers:

- data ingestion
- ETL
- cataloging and governance
- concise exploratory analysis
- feature extraction

The pipeline is designed to be reproducible and deployed on Azure.

---

## Dataset
- **Dataset name:** eCommerce Behavior Data from Multi-Category Store
- **Source:** Kaggle
- **Phase 1 file used:** `2019-Oct.csv`
- **Approximate dataset characteristics:** monthly event-log CSV files from Oct 2019 to Apr 2020
- **Relevant fields used in this phase:**
  - `event_time`
  - `event_type`
  - `user_id`
  - `user_session`

This dataset is suitable because it contains timestamped user actions that allow both time-based analysis and sequential funnel analysis.

---

## Project Hypothesis
### 1. Peak Time
User activity is not evenly distributed across the week. We expect clear peak periods where the volume of views is highest by **day of week** and **hour of day**.

### 2. Purchase Drop-off in the Sequential Funnel
In the sequential funnel **View → Cart → Purchase**, we expect the largest drop-off to happen at the **View → Cart** stage more than the **Cart → Purchase** stage.

Only sequential behavior is considered. Sessions where a purchase happens directly after a view without a cart step are excluded from the funnel logic.

---

## Azure Services Used
The implementation uses the following Azure services:

- **Azure Storage Account** for raw and processed data storage
- **Azure Data Factory** for ingestion pipeline execution
- **Azure Databricks** for ETL, feature extraction, exploratory analysis, and output generation

---

## Storage Layout
The data is organized into zones following the raw/processed structure required for Phase 1.

### Raw Zone
- `raw/ecommerce-behavior/2019-Oct.csv`

### Processed Zone
- `processed/ecommerce-behavior/processed_2019_oct.csv`
- `processed/ecommerce-behavior/clean_events/`
- `processed/ecommerce-behavior/peak_time_metrics/`
- `processed/ecommerce-behavior/sequential_funnel_metrics/`
- `processed/ecommerce-behavior/data_quality_summary/`

This structure preserves the original raw data while storing transformed analytical outputs separately.

---

## Data Ingestion
The ingestion design is:

- **Ingestion mode:** batch
- **Input format:** CSV
- **Refresh strategy:** manual batch upload for Phase 1
- **Raw data preservation:** the original source file is stored in the raw zone without modification

### Ingestion Pipeline
Azure Data Factory was used to create a pipeline that copies the raw CSV into the processed area as an initial ingestion artifact.

This provides:
- a reproducible ingestion step
- raw data preservation
- a clear Azure pipeline artifact for deployment evidence

---

## ETL Process
Azure Databricks was used for ETL and analytical transformation.

### ETL steps implemented
1. Read the raw CSV from Azure Data Lake Storage Gen2
2. Keep the columns relevant to the project
3. Remove exact duplicate rows
4. Drop rows missing essential fields:
   - `event_time`
   - `event_type`
   - `user_id`
   - `user_session`
5. Keep only relevant event types:
   - `view`
   - `cart`
   - `purchase`
6. Derive project features from `event_time`
7. Build analytical outputs for peak-time and funnel analysis
8. Write cleaned and aggregated results to the processed zone

This ETL process is reproducible because the transformations are explicitly defined in the Databricks notebook and output paths are fixed.

---

## Cataloging and Governance
The project documents the dataset through a metadata-oriented structure rather than a separate external catalog service.

### Schema Overview
Main fields used in this phase:

- `event_time` → timestamp
- `event_type` → string
- `product_id` → integer
- `category_id` → long
- `category_code` → string
- `brand` → string
- `price` → double
- `user_id` → integer
- `user_session` → string
- `day_of_week` → string
- `hour_of_day` → integer

### Basic Lineage
Raw source file  
→ Azure Storage raw zone  
→ ADF ingestion pipeline  
→ Databricks ETL and feature extraction  
→ processed analytical outputs in Azure Storage

### Assumptions
- Only `view`, `cart`, and `purchase` are relevant for funnel analysis
- Sequential funnel logic requires ordered progression within the same `user_session`
- Direct view-to-purchase cases without cart are excluded from the sequential funnel

---

## Exploratory Data Analysis
The exploratory analysis was intentionally concise, as required in the brief.

### Checks performed
- event type distribution
- basic price statistics
- duplicate count
- null summary for key fields
- outlier-risk check on `price`

### Purpose
The objective of this EDA was not exhaustive business reporting. It was to assess whether the dataset is ready for downstream analytics and to identify possible risks such as duplicates, missing essential fields, and price outliers.

---

## Feature Extraction
Two initial features were extracted directly from `event_time`:

- **`day_of_week`**
- **`hour_of_day`**

### Justification
These features are directly aligned with the project hypothesis because peak time is defined by day and hour. They are also simple, interpretable, and reproducible.

---

## Peak-Time Analysis
To evaluate peak activity, the pipeline:

1. filters the cleaned data to `event_type = "view"`
2. groups records by:
   - `day_of_week`
   - `hour_of_day`
3. computes `view_count`
4. sorts the grouped results in descending order of view volume

This produces a table that identifies the time periods with the highest user activity.

## Azure Outputs
The following outputs were generated in the processed zone:

- cleaned event-level dataset
- peak-time metrics
- sequential funnel metrics
- data quality summary
- initial ADF copied processed CSV artifact

