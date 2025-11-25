# Credit Card Transaction Analysis and Feature Engineering

A comprehensive exploratory data analysis (EDA) and feature engineering project based on 788,521 real-world government credit card transactions (2018–2023).
This project focuses on understanding spending behaviors, detecting irregular patterns, and building a rich feature set for downstream anomaly-detection models.

---

## 1. Project Overview

This repository contains the full analysis and engineered datasets for a large-scale credit card transaction dataset.
The primary objectives are:

1. Clean and standardize raw transactional data
2. Perform extensive univariate and multivariate EDA
3. Identify spending patterns across departments, divisions, merchants and dates
4. Engineer high-signal features suitable for unsupervised anomaly detection
5. Build modular and reproducible code that can be extended to ML modeling

The analysis is implemented in Python using Jupyter Notebook.

---

## 2. Dataset Description

| Column           | Type              | Description                                |
| ---------------- | ----------------- | ------------------------------------------ |
| Year             | int               | Reported fiscal year                       |
| Month            | int               | Fiscal month                               |
| Department       | string            | Government department making the purchase  |
| Division         | string            | Sub-unit within department                 |
| Merchant         | string            | Vendor or service provider                 |
| TranxDescription | string            | MCC category or description                |
| TranxDate        | string → datetime | Mixed-format date field                    |
| TrnxAmount       | float             | Transaction amount (positive and negative) |

Total rows: **788,521**
Time range covered: **2018–2023**

---

## 3. Data Cleaning

Key data cleaning steps included:

* Standardized mixed date formats (MM/DD/YYYY and MM-DD-YYYY)
* Removed whitespace and formatting inconsistencies
* Converted dates into pandas datetime format
* Ensured chronological sorting within organizational units
* Verified no missing essential fields after processing

All transformations are performed with reproducible code in the notebook.

---

## 4. Exploratory Data Analysis (EDA)

### 4.1 Univariate Analysis

* Transaction amounts exhibit strong right-skewness
* Refunds appear as negative values
* Merchant and department frequencies follow a long-tail distribution
* Seasonal effects are observed across months and years

Visualization techniques include:

* Log-transformed histograms
* Boxplots for outlier detection
* Frequency plots for categorical variables

### 4.2 Bivariate & Multivariate Analysis

* Boxplots reveal department-level spending differences
* Monthly aggregated spending shows clear seasonal patterns
* Merchant-level variability highlights high-volume vendors and spending clusters

These analyses provide insight into normal vs. abnormal spending patterns.

---

## 5. Feature Engineering

A major focus of the project is building a feature set optimized for anomaly detection.
The following categories were engineered:

### 5.1 Time-Based Features

* DayOfWeek
* IsWeekend
* DayOfMonth
* WeekOfMonth
* Quarter
* IsMonthStart / IsMonthEnd
* TimeGap between transactions within each department

### 5.2 Amount-Based Features

* Log-transformed amounts
* Z-scored amount values
* Relative-to-department spending
* Relative-to-merchant spending

### 5.3 Behavior-Based Features

* Merchant transaction frequency
* Department–merchant joint frequency
* Transaction description frequency

### 5.4 Pattern Features

* ShortIntervalFlag: rapid consecutive spending
* AmountDiff: change relative to previous transaction
* SameDayTotalAmount per department

These features capture temporal, behavioral and contextual signals necessary for anomaly detection and spending behavior modeling.

---

## 6. Repository Structure

```
Credit-Card-Transaction-EDA/
│
├── CreditCardEDA.ipynb      # Main notebook with full analysis and feature engineering
├── README.md                # Project documentation
└── data/                    # (Optional) Input dataset or processed subsets
```

---

## 7. Technical Stack

* Python 3
* Pandas
* NumPy
* Seaborn
* Matplotlib
* Jupyter Notebook

All code is reproducible and written with clear explanations and comments.

---

## 8. Key Insights and Findings

* Spending is highly concentrated in a small number of departments and vendors
* There is strong seasonality and weekly behavior in transaction patterns
* Departments show distinct spending profiles and frequencies
* Log-transformation improves interpretability of skewed values
* Engineered features expose patterns not visible in raw data, such as burst spending, abnormal merchant counts, and within-department irregularities

These insights lay the groundwork for building anomaly-detection algorithms.

---

## 9. Future Work

The next stages of the project will include:

1. Unsupervised anomaly detection (Isolation Forest, LOF, Autoencoders)
2. Entity-level profiling (merchant, department, user-level risk segmentation)
3. Pattern classification using clustering (K-means, HDBSCAN)
4. Model evaluation using reconstruction errors and distance metrics
5. Dashboarding and interactive visualization

---

## 10. How to Run

1. Clone the repository

```bash
git clone https://github.com/RENEGADES20/Credit-Card-Transaction-EDA.git
```

2. Open the notebook

```bash
jupyter notebook CreditCardEDA.ipynb
```

3. Install dependencies if needed

```bash
pip install pandas numpy seaborn matplotlib
```

---


需要哪个？
