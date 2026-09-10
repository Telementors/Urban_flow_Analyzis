Here is the updated `README.md` markdown tailored specifically to your project structure (`Urban_Mobility_Analytics`), including your `dashboards/power_bi`, single `Preprocessing_pipeline.ipynb` notebook, and `observability` directory:

```markdown
# Urban Mobility Analytics — Datathon Challenge

An end-to-end Data Engineering, Machine Learning, and Observability pipeline for processing, cleaning, feature engineering, and visualizing high-volume urban transit flow datasets (~48.6+ million records / 5GB+).

---

## 📌 Project Overview

This project focuses on handling multi-gigabyte monthly urban transit data, engineering domain-specific features, applying temporal rolling-window validation splits, and rendering analytical insights via Power BI dashboards and observability monitoring.

---

## 🛠️ Data Pipeline & Architecture

To process ~48.6 million records across 12 monthly CSV files without triggering RAM out-of-memory crashes, the pipeline relies on **chunked streaming execution** (`chunksize=100,000`).


```

Raw Monthly CSVs (2025-04 to 2026-03)
│
▼

1. Preprocessing Pipeline (`Preprocessing_pipeline.ipynb`)
├── Monthly Consolidation & Data Cleaning
├── Zone Lookup Encodings (One-Hot & Frequency Encoding)
├── Indicator Feature Engineering (`is_airport_trip`, `offline_record_flag`)
├── Multicollinearity Reduction (Dropping `charge_total`, `base_fare`)
└── Rolling Window Time-Series Splitting (3-Train / 1-Test)
│
▼
2. Incremental Feature Scaling
└── `StandardScaler` via `partial_fit` (Fitted on Train Set Only)
│
├──► Cleaned & Scaled Datasets (`train_cleaned.csv` / `test_cleaned.csv`)
│
├──► Power BI Dashboards (`dashboards/power_bi/`)
│
└──► Pipeline Observability & Logging (`observability/`)

```

---

## 📂 Repository Structure


```

Urban_Mobility_Analytics/
├── dashboards/
│   └── power_bi/                     # Power BI reports & dashboard templates
├── notebooks/
│   └── Preprocessing_pipeline.ipynb  # Main end-to-end data processing & engineering notebook
├── observability/                     # Pipeline logs, data quality checks, & metrics tracking
├── .gitignore
├── README.md                          # Project documentation
└── requirements.txt                   # Python dependencies

```

---

## ⚙️ Data Preprocessing & Feature Engineering Highlights

All data transformation, encoding, feature engineering, and scaling operations are encapsulated within `notebooks/Preprocessing_pipeline.ipynb`:

1. **Zone Metadata Encodings:**
   * **One-Hot Encoding:** Applied to low-cardinality categorical variables (`borough_name`, `service_zone`).
   * **Frequency Encoding:** Applied to high-cardinality zone categories (`zone_name`).
   * **Zone Mapping:** Merged with `origin_loc_id` and `dest_loc_id`.

2. **Feature Engineering:**
   * **`is_airport_trip`:** Derived binary indicator ($1$ if $\text{Airport\_fee} > 0$, else $0$).
   * **`offline_record_flag`:** Encoded to binary values ($0$ for 'N', $1$ for 'Y').

3. **Multicollinearity & Noise Reduction:**
   * Removed highly redundant features ($r > 0.90$) identified via correlation heatmaps:
     * Dropped `charge_total` (perfect collinearity with sum of fare components).
     * Dropped `base_fare` (redundant with `distance_miles`, $r = 0.94$).
     * Dropped zero-variance / constant features (`offline_record_flag` post-filtering).
   * Filtered out invalid records (`base_fare <= 0`, `distance_miles <= 0`).

4. **Incremental Scaling:**
   * **`StandardScaler`** fit strictly on Training data using `partial_fit` to prevent data leakage, then applied (`transform`) to Test sets.

---

## ⏱️ Validation Strategy: Time-Series Rolling Window

To prevent temporal data leakage, a **3-Month Train / 1-Month Test Rolling Window** scheme was adopted across the 12 monthly files:

| Batch | Training Months (3 Months) | Test Month (1 Month) |
| :--- | :--- | :--- |
| **Set 1** | 2025-04, 2025-05, 2025-06 | 2025-07 |
| **Set 2** | 2025-08, 2025-09, 2025-10 | 2025-11 |
| **Set 3** | 2025-12, 2026-01, 2026-02 | 2026-03 |

---

## 🚀 How to Run

1. **Clone the Repository & Install Dependencies:**
   ```bash
   git clone <your-repository-url>
   cd Urban_Mobility_Analytics
   pip install -r requirements.txt

```

2. **Run the Preprocessing Pipeline:**
Open and execute `notebooks/Preprocessing_pipeline.ipynb` in Google Colab or your local Jupyter environment.
3. **Power BI Dashboards:**
Navigate to `dashboards/power_bi/` to connect the processed CSV outputs into Power BI for interactive analytics.

```

```