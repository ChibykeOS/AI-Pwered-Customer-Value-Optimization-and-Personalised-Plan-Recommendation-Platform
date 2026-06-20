# AI-Powered Customer Value Optimization & Personalized Plan Recommendation Platform

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-98.52%25%20Acc-green.svg?style=for-the-badge&logo=lightgbm&logoColor=white)](https://github.com/microsoft/LightGBM)
[![XGBoost](https://img.shields.io/badge/XGBoost-98.37%25%20Acc-orange.svg?style=for-the-badge&logo=xgboost&logoColor=white)](https://github.com/dmlc/xgboost)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An end-to-end Machine Learning-driven solution designed for **Company A** to optimize subscriber plan alignments, mitigate "bill shock" churn, and unlock significant recurring revenue streams. Based on a subscriber base of 100,000, this platform delivers an estimated **$2.24 Million Net Annual Value Lift**.

---

## 📌 Executive Summary

Modern telecom operators face high customer acquisition costs (averaging **$300/subscriber**) and high voluntary churn rates (accounting for **85% of telecom customer loss**). "Bill shock" from unpredictable overage charges is a primary churn driver. 

This project implements a **Personalized Plan Recommendation Engine** that utilizes subscriber demographics, device properties, and historical usage trends to predict the optimal plan segment for each customer. By proactively transitioning subscribers to right-sized plans, Company A can capture recurring revenue, increase customer lifetime value (LTV), and drastically reduce voluntary churn.

---

## 📊 Exploratory Data Analysis (EDA) Insights

Analysis of the merged `Client.csv` and `Record.csv` datasets (100,000 subscribers, 100 features) revealed significant pain points:

*   **Overage Volatility**: **57.09% of subscribers** pay overages monthly, averaging **$23.67/month**. Overage revenues contribute **23.09% ($1.35M/month)** of total monthly revenue, but represent highly volatile revenue and cause major customer dissatisfaction.
*   **Plan Misalignment**:
    *   **Underutilizers (1.73% of subscribers)**: Pay high monthly recurring charges (MRC) (mean: $69.83) but use minimal minutes (mean: 81.7 mins). These are high-risk candidates for churn.
    *   **Overutilizers (14.73% of subscribers)**: Pay low MRC but incur substantial monthly overage charges (averaging **$35.38** in overages). These are prime candidates for upsells.
*   **Usage Dynamics**: **78.36% of subscribers** experience month-over-month usage fluctuations **exceeding 20%** compared to their 3-month average, rendering static sign-up plans obsolete.

---

## 🤖 Machine Learning Formulation

To prevent target leakage, all direct usage and revenue proxies (such as current month minutes and overage revenues) are dropped during training. The models must predict the optimal plan tier using only demographics, equipment metrics, and historical 3-month/6-month averages.

### Target Plan Tiers
| Tier | Plan Name | Criteria |
| :--- | :--- | :--- |
| **0** | **Economy Plan** | Usage $\le$ 150 minutes and overages < $10 |
| **1** | **Standard Plan** | Usage 150–600 minutes and overages < $10 |
| **2** | **Premium Plan** | Usage 600–1200 minutes, OR usage $\le$ 600 with overages $10–$20 |
| **3** | **Unlimited Plan** | Usage > 1200 minutes, OR overages $\ge$ $20 (High-risk "Bill Shock" cohort) |

### Feature Engineering
The script engineers several business-centric ratios to boost model performance:
*   `overage_ratio`: Overage revenue divided by total revenue.
*   `revenue_per_minute`: Total revenue divided by total minutes of use.
*   `dropped_call_ratio`: Total dropped voice/data calls divided by total attempted calls.
*   `completed_call_ratio`: Completed calls divided by total attempted calls.

### Model Evaluation
Both models were evaluated on an 80/20 stratified train-test split:

| Model | Test Accuracy | Cohen's Kappa | Macro-F1 |
| :--- | :--- | :--- | :--- |
| **LightGBM Classifier (Best)** | **98.52%** | **98.00%** | **98.00%** |
| **XGBoost Classifier** | **98.37%** | **97.80%** | **97.00%** |

> [!TIP]
> **LightGBM** was selected as the final production model due to superior generalization and faster inference speeds on tabular data. Its hyperparameters utilize balanced class weighting to account for target class distribution imbalances.

---

## 📈 Quantified Business Value

Our proposed retention and upsell campaign targets two critical subscriber cohorts:

### Cohort 1: Overage Mitigation (Upsell to Unlimited)
*   **Target Size**: 23,826 overage-heavy subscribers.
*   **Action**: Proactive migration to a flat Unlimited Plan ($65/mo flat vs. current $45/mo MRC).
*   **Revenue Lift**: **+$119,130/month** (average MRC increase offset by overage reduction).
*   **LTV Protection**: 25% churn reduction saves **1,191 subscribers/year**, representing **+$29,782/month** in saved Customer Acquisition Costs (CAC @ $300).

### Cohort 2: Proactive Downgrades (Churn Prevention)
*   **Target Size**: 1,730 underutilizing subscribers.
*   **Action**: Proactive downgrade recommendation to Standard Plan ($45/mo vs. current $69.83 MRC).
*   **Short-term MRC Impact**: **-$42,955/month**.
*   **LTV Protection**: 50% churn reduction saves **692 subscribers/year**, representing **+$20,760/month** in saved CAC.

### 💰 Net Financial Impact
*   **Net Monthly Value Lift**: **+$186,281/month**
*   **Net Annual Value Realized**: **+$2.24 Million**

---

## 📁 Repository Structure

```
├── Client.csv                            # Customer demographic and account information (100k rows)
├── Record.csv                            # Usage history, network performance, and billing logs (100k rows)
├── train_model.py                        # Python training script for model training and evaluation
├── final_assignment.ipynb                # End-to-end Jupyter Notebook (EDA, training, visualization)
├── best_plan_recommendation_model.pkl    # Serialized LightGBM model artifact
├── feature_importances.csv               # Ranked predictive drivers from the LightGBM model
├── confusion_matrix.npy                  # Saved evaluation confusion matrix
├── eda_findings.md                       # Comprehensive markdown summary of EDA insights
├── presentation_slides_draft.md          # 12-slide presentation draft for executive presentation
├── ENG_Company _A_ Dataset Overview.docx  # Raw data dictionary and context provided by Company A
└── README.md                             # This overview documentation
```

---

## 🚀 Getting Started & Execution

### Prerequisites
Install all model dependencies:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm joblib
```

### Model Training & Evaluation
To retrain the model, evaluate performance, and generate the predictions/feature importances:
```bash
python train_model.py
```

Upon execution, the script will:
1. Merge the demographic and record datasets.
2. Clean missing numeric/categorical values.
3. Engineer business ratios and drop target leakage columns.
4. Train both LightGBM and XGBoost with balanced class weights.
5. Save the best-performing model to `best_plan_recommendation_model.pkl` and output feature importance logs.
