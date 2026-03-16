# Dataset Metadata – E-Commerce Behavior (Oct 2019)

## Dataset Overview

Dataset: E-commerce Customer Behavior Dataset  
Source: Kaggle  
File Used: 2019-Oct.csv  

The dataset contains user interaction events from an online e-commerce platform including product views, cart additions, and purchases.

Raw dataset is stored in Azure Data Lake Gen2 under the **raw container**, and processed outputs are stored in the **processed container**.

---

# Schema Definition

| Column | Data Type | Description |
|------|------|------|
| event_time | timestamp | Timestamp of the user event |
| event_type | string | Type of interaction (view, cart, purchase) |
| product_id | integer | Unique identifier of the product |
| category_id | long | Identifier for the product category |
| category_code | string | Hierarchical category name |
| brand | string | Product brand |
| price | double | Product price |
| user_id | integer | Unique identifier of the user |
| user_session | string | Session identifier |

---

# Derived Features

Additional features were created during preprocessing:

| Feature | Description |
|------|------|
| day_of_week | Day extracted from event_time |
| hour_of_day | Hour extracted from event_time |

---

# Data Pipeline Lineage

Data flows through the following architecture:
Kaggle Dataset
>
Azure Data Lake Storage Gen2 (raw container)
>
Azure Data Factory pipeline
>
Databricks ETL Notebook
>
Processed Data Lake outputs


Generated outputs include:

- Clean events dataset
- Peak time metrics
- Sequential funnel metrics
- Data quality summary

---

# Assumptions

1. Only three event types are considered meaningful for behavioral analysis:  
   **view → cart → purchase**

2. Funnel analysis assumes chronological order:
     view_time < cart_time < purchase_time
3. The dataset is treated as a **monthly snapshot** and not incrementally updated.

---

# Versioning

Dataset Version: 2019 October  
Storage Path: raw/ecommerce-behavior/2019-Oct.csv
