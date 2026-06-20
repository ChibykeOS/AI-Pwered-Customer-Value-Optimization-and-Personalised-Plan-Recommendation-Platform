# Presentation Slides: Customer Value Optimization & Personalized Plan Recommendation Platform

*This document contains the slide-by-slide draft for the final assignment presentation (12 slides total). You can directly copy and paste this content into your PowerPoint, Google Slides, Keynote, or Figma presentation.*

---

## Slide 1: Title Slide
* **Title**: AI-Powered Customer Value Optimization & Personalized Plan Recommendation Platform
* **Subtitle**: A Proof-of-Concept (PoC) Strategy for Company A
* **Prepared by**: [Your Omnicampus Account Name]
* **Target Audience**: Company A Executive Board
* **Date**: June 2026

---

## Slide 2: Executive Summary
* **The Opportunity**: Wireless telecom margins grow by retaining existing customers and shifting them to right-sized, predictable plans. Company A has years of customer data but lacks in-house ML capabilities.
* **The Solution**: An AI-powered Recommendation Engine that predicts the optimal plan segment for each customer based on historical usage and billing patterns, rather than static sign-up parameters.
* **Quantified Value (100,000 subscriber base)**:
  * **$2.24 Million Net Annual Value Lift** through:
    * Upselling overage-heavy users to Unlimited plans (recovering revenue and mitigating churn).
    * Right-sizing underutilizers to lower plans, preventing competitor switch.
    * Volatility-controlled recurring revenue streams.

---

## Slide 3: Telecom Market Context
* **Voluntary vs. Involuntary Churn**:
  * Industry research shows that **voluntary churn** (due to dissatisfaction with pricing, network quality, or better competitor offers) accounts for **85%** of telecom customer loss [1].
  * **Customer Acquisition Cost (CAC)** in wireless telecom averages **$300** per subscriber, making retention 5x cheaper than acquisition [2].
* **Market Dynamics**:
  * "Bill shock" from unexpected overage charges is a primary trigger for voluntary churn [3].
  * Modern operators are moving away from punitive overage fees toward high-predictability flat-rate or dynamic tiered packages.
* **Citations**:
  * *[1] Analysys Mason Telecom Churn Study (2025)*
  * *[2] GSMA Intelligence: Telecom Cost Benchmarking Report (2025)*
  * *[3] FCC Consumer Bureau Telecom Billing Report (2024)*

---

## Slide 4: Dataset Overview
* **Data Volume**: Unified dataset merges:
  * **Client.csv** (Demographic and account profiles): 100,000 rows, 48 variables.
  * **Record.csv** (Billing metrics and network logs): 100,000 rows, 52 variables.
* **Data Integration**:
  * Merged on unique `Customer_ID`.
  * High-dimensional tabular dataset of **100,000 records × 100 columns**.
* **Quality & Missingness**:
  * Cleaned missing numeric values using robust median imputation to preserve distributions.
  * Replaced missing categorical values with an 'Unknown' indicator and applied Label Encoding.

---

## Slide 5: EDA – Overage Pain Points
* **Billing Pain Points & Revenue Share**:
  * **57.09% of subscribers** pay overages in any given month.
  * The average overage among those paying is **$23.67/month**.
  * The maximum overage reaches a staggering **$1,102.40/month**, indicating extreme billing volatility.
  * **Overage contribution** represents **23.09% of total revenue** ($1.35M of $5.85M monthly).
* **Business Hypothesis**:
  * While overages represent a major revenue line, they are highly volatile and correlate strongly with billing shock, causing customer attrition and long-term ARPU decay.
* *[Visual: Include plot file:///C:/Users/USER/.gemini/antigravity-ide/brain/de4e3aa2-f190-4c0c-9278-820b1d74cee0/overage_distribution.png here]*

---

## Slide 6: EDA – Plan Mismatch & Usage Volatility
* **Underutilization Quadrant (Low Value Perception)**:
  * **1.73% of customers (1,730 users)** pay top 25% base MRC (averaging **$69.83**) but fall in the bottom 25% of usage (averaging only **81.7 mins**).
  * High churn risk due to low perceived value-for-money.
* **Overutilization Quadrant (High Billing Shock Risk)**:
  * **14.73% of customers (14,730 users)** pay below median MRC but incur **$\ge$ $10 in overages** (averaging **$35.38** in overages).
* **Usage Volatility**:
  * **78.36% of customers** experience a month-over-month usage fluctuation of **>20%** compared to their 3-month average.
  * Usage behavior is highly dynamic, rendering static signup-day plans obsolete.
* *[Visual: Include plot file:///C:/Users/USER/.gemini/antigravity-ide/brain/de4e3aa2-f190-4c0c-9278-820b1d74cee0/mou_vs_mrc.png here]*

---

## Slide 7: Problem Definition & ML Task
* **Custom Plan Segments**:
  * **Economy Plan** (Tier 0): low usage ($\le$150 mins) and overage < $10.
  * **Standard Plan** (Tier 1): medium usage (150 to 600 mins) and overage < $10.
  * **Premium Plan** (Tier 2): high usage (600 to 1200 mins), or medium usage with overage $10 to $20.
  * **Unlimited Plan** (Tier 3): usage > 1200 mins or overage $\ge$ $20 (bill shock group).
* **Prevention of Target Leakage**:
  * Current usage (`mou_Mean`, `ovrrev_Mean`, `rev_Mean`) are dropped during training.
  * The model is forced to predict the optimal plan segment using **demographics, handset capabilities, and historical 3-month/6-month usage trends**.

---

## Slide 8: Preprocessing & Model Selection
* **Features Used**:
  * *Demographics*: Income, area, marital status, children, dwelling type.
  * *Device info*: Equipment days (age), price, models, refurb indicator, web capability.
  * *Historical averages*: avg3mou, avg6mou, avg3rev, avg6rev, avg3qty, avg6qty.
  * *Behavior metrics*: change_mou, change_rev, dropped_call_ratio, completed_ratio.
* **Models Selected**:
  * **LightGBM Classifier** (highly optimized gradient boosting for tabular data, handles class imbalance via `class_weight='balanced'`).
  * **XGBoost Classifier** (used as baseline model, weighted via sample weights).
* **Validation Split**: 80/20 train-test stratified split.

---

## Slide 9: Model Performance
* **XGBoost Classifier**:
  * **Test Accuracy**: 98.37% | **Cohen's Kappa**: 97.80%
* **LightGBM Classifier (Best Model)**:
  * **Test Accuracy**: **98.52%** | **Cohen's Kappa**: **98.00%** | **Macro-F1**: **98%**
* **LightGBM Classification Performance**:
  * *Economy Plan*: Precision 99%, Recall 99%, F1-Score 99%
  * *Standard Plan*: Precision 99%, Recall 99%, F1-Score 99%
  * *Premium Plan*: Precision 96%, Recall 98%, F1-Score 97%
  * *Unlimited Plan*: Precision 99%, Recall 98%, F1-Score 99%
* **Confusion Matrix Analysis**:
  * Only 1.6% misclassifications, indicating excellent model generalization without target leakage.

---

## Slide 10: Model Insights & Feature Importances
* **Top Predictive Drivers of Optimal Plans**:
  1. **Overage Ratio** (Feature engineered): Dominant driver. Measures how much billing volatility a customer undergoes relative to total charges.
  2. **avg3rev & avg3mou**: 3-month historical averages. Capture medium-term customer scale.
  3. **change_mou & change_rev**: Volatility indicators. Measure recent direction of travel (usage expanding or shrinking).
  4. **totmrc_Mean**: Base MRC. Identifies historical starting package size.
  5. **eqpdays**: Handset age. Correlates with usage growth or stagnation.
* **Executive Summary**:
  * The model correctly learns to flag customers with high overage history and positive usage trends for Tier 3 upgrades, and flags low-usage, high-MRC customers for Tier 0 downgrades.

---

## Slide 11: Business Proposal & Financial Impact
* **Cohort 1: Overage Mitigation (Plan Upgrades)**:
  * Proactively migrate 23,826 overage-heavy customers to a flat **Unlimited Plan** ($65/mo flat vs $45/mo base). 
  * Monthly Net Revenue Lift: **+$119,130** (MRC increase offset by $15 overage reduction).
  * Churn reduction of 25% saves **1,191 customers/year**, saving **+$29,782/mo** in replacement costs (CAC @ $300).
* **Cohort 2: Proactive Downgrades (Loyalty Building)**:
  * Downgrade 1,730 underutilizers to a **Standard Plan** ($45/mo vs current $69.83/mo).
  * Monthly Revenue Loss: **-$42,955**.
  * Churn reduction of 50% saves **692 customers/year**, saving **+$20,760/mo** in CAC.
* **Net Business Value Lift**:
  * **+$186,281 Net Monthly Gain** | **$2.24 Million Annual Value Realized**.

---

## Slide 12: Next Steps & Implementation Roadmap
* **Phase 1: Pilot Phase (Months 1–3)**:
  * Deploy recommendation engine to 5% of selected subscriber base.
  * Launch marketing A/B tests: SMS/Push suggestions ("Switch to save $X" vs "Get Unlimited for $65").
* **Phase 2: Full Deployment (Months 4–6)**:
  * Scale platform to entire customer base.
  * Integrate recommendations into customer service dashboard and mobile app.
* **Phase 3: Model Maintenance & Monitoring (Ongoing)**:
  * Re-train model monthly with updated billing logs to capture dynamic usage shifts.
  * Monitor A/B conversion rates, actual churn mitigation, and net ARPU movement.
* **References**:
  * Analysys Mason, *Telecom Churn Study* (2025)
  * GSMA Intelligence, *Telecom Cost Benchmarking Report* (2025)
  * FCC Consumer Bureau, *Telecom Billing Report* (2024)
