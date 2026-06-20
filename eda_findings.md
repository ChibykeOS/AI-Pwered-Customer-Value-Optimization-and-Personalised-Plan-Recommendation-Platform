# Exploratory Data Analysis (EDA) Findings: Customer Plan Optimization

To validate the business proposal, we analyzed the merged customer demographic (`Client.csv`) and usage (`Record.csv`) datasets. The data strongly supports building an **AI-Powered Customer Value Optimization & Personalized Plan Recommendation Platform**.

Here is the empirical evidence:

---

## 1. Overage Pain Points (Bill Shock & Customer Experience)
Overage charges represent both a significant source of customer dissatisfaction (due to unpredictable "bill shock") and a major portion of the client's current revenue structure.

* **57.09% of customers pay overage charges** in any given month.
* The average overage charge among those who pay is **$23.67/month**.
* The maximum overage charge recorded is a staggering **$1,102.40/month**, indicating extreme billing anomalies.
* **Overage Contribution**: Overage revenue constitutes **23.09% of total monthly revenue** ($1.35M out of $5.85M total). 

> [!WARNING]
> While overage fees are a major revenue source ($1.35M/month), they are highly volatile and correlate strongly with customer dissatisfaction and eventual churn. Migrating these customers to predictable, right-sized plans protects future recurring revenue.

![Overage Revenue Distribution](file:///C:/Users/USER/.gemini/antigravity-ide/brain/de4e3aa2-f190-4c0c-9278-820b1d74cee0/overage_distribution.png)

---

## 2. Plan Mismatch: Underutilization vs. Overutilization
The analysis reveals substantial misalignment between what customers pay in monthly recurring charges (MRC) and what they actually use.

* **Overutilized Accounts (14.73% of customers)**:
  * These customers pay a monthly recurring charge *below the median* but incur **at least $10 in overage charges** (averaging **$35.38** in overages).
  * Their mean monthly usage is **565.4 minutes** (higher than the general population average of 513.6).
  * **Opportunity**: These 14,730 customers are prime candidates for an **upsell** to higher-tier, higher-MRC plans that include more minutes. This increases predictable recurring revenue (reducing volatility) while saving the customer from excessive overage fees.
* **Underutilized Accounts (1.73% of customers)**:
  * These customers pay a high monthly recurring charge (top 25%, averaging **$69.83**) but use very few minutes (bottom 25%, averaging only **81.7 minutes**).
  * **Opportunity**: These 1,730 customers represent a severe churn risk due to a low value-for-money perception. Proactively recommending a **plan downgrade** (or standard value plan) increases trust, builds brand loyalty, and prevents them from switching to a competitor.

![Minutes of Use vs. Monthly Recurring Charge](file:///C:/Users/USER/.gemini/antigravity-ide/brain/de4e3aa2-f190-4c0c-9278-820b1d74cee0/mou_vs_mrc.png)

---

## 3. High Usage Volatility
Static plans are insufficient because customer telecommunications behavior changes dynamically.

* **78.36% of customers experience a month-over-month usage fluctuation of greater than 20%** (relative to their previous 3-month average).
* The **average absolute percentage change in usage is 148.05%**.
* This massive volatility means that a plan chosen at sign-up is highly unlikely to remain optimal for the customer over time.

> [!TIP]
> Because customer usage behavior fluctuates so heavily month-to-month, a static plan assignment fails. A dynamic, **AI-driven plan recommendation engine** that re-evaluates customer usage patterns monthly is the only way to maintain alignment.

![Overage Revenue vs. Monthly Recurring Charge](file:///C:/Users/USER/.gemini/antigravity-ide/brain/de4e3aa2-f190-4c0c-9278-820b1d74cee0/mrc_vs_overage.png)

---

## Conclusion
The data validates our business case:
1. **Overage is widespread and volatile** (57% of customers affected).
2. **Clear plan misalignment exists** (14.7% overpaying via overages, 1.7% overpaying via base MRC).
3. **Usage is highly volatile** (78% of customers fluctuating >20% monthly), which necessitates a dynamic, automated AI optimizer rather than manual plan changes.
