# 👥 Employee Attrition Prediction — HR Analytics

<p align="center">
  <a href="https://colab.research.google.com/github/GIVEN-CHINYAMA/Employee-Attrition-Prediction/blob/main/Employee_Attrition_Prediction_HR_Analytics.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="28"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/XGBoost-Gradient%20Boosting-FF6600?logo=xgboost" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-8A2BE2" alt="SHAP"/>
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Status-Complete-2ecc71" alt="Status"/>
</p>

<p align="center">
  <strong>An end-to-end machine learning pipeline that predicts employee resignation risk, identifies key attrition drivers using SHAP explainability, and quantifies financial exposure — enabling HR teams to intervene before it is too late.</strong>
</p>

---

## 📌 Table of Contents        

- [Overview](#-overview)
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Models & Results](#-models--results)
- [Key Findings](#-key-findings)
- [HR Recommendations](#-actionable-hr-recommendations)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technologies](#-technologies)
- [Author](#-author)

---

## 🔍 Overview

Employee attrition — the voluntary resignation of staff — is one of the most expensive and disruptive challenges organisations face. Replacing a single employee costs between **50% and 200% of their annual salary** once recruitment, onboarding, training, and lost productivity are factored in.

This project builds a **production-grade HR analytics pipeline** that:

- Predicts resignation risk at the **individual employee level**
- Identifies the **top drivers of attrition** across the workforce using SHAP
- Segments employees into **four risk tiers** for targeted interventions
- Quantifies **financial exposure** from predicted attrition by department
- Provides **plain-language explanations** for every prediction — legally and ethically compliant

---

## 🎯 Business Problem

> *"Which employees are most likely to resign, and what organisational factors are driving attrition — so that HR can act before it is too late?"*

| Objective | Deliverable |
|---|---|
| Predict individual resignation risk | Probability score (0–100%) per employee |
| Identify workforce attrition drivers | SHAP global & local feature importance |
| Segment workforce by risk level | 4-tier risk classification system |
| Quantify financial exposure | Replacement cost model by department |
| Provide HR-interpretable explanations | SHAP waterfall plots per employee |

---

## 📊 Dataset

**IBM HR Analytics Employee Attrition Dataset**

| Property | Value |
|---|---|
| Source | IBM / Kaggle HR Analytics |
| Employees | 1,470 |
| Features | 35 (demographics, compensation, satisfaction, tenure) |
| Target | `Attrition` — Yes / No |
| Class Imbalance | ~84% Stayed · ~16% Left |
| Missing Values | None |

The dataset is loaded automatically in the notebook — no manual download is required. It covers employee demographics, job role and level, monthly income, satisfaction scores (job, environment, work-life balance, job involvement), business travel frequency, overtime status, and tenure metrics.

---

## 🗂️ Project Structure

```
Employee-Attrition-Prediction/
│
├── Employee_Attrition_Prediction_HR_Analytics.ipynb   ← Main notebook
├── README.md
│
└── outputs/  (generated during notebook execution)
    ├── attrition_distribution.png
    ├── numerical_distributions.png
    ├── categorical_attrition.png
    ├── satisfaction_attrition.png
    ├── correlation_heatmap.png
    ├── smote_balance.png
    ├── roc_pr_curves.png
    ├── confusion_matrices.png
    ├── shap_summary.png
    ├── shap_bar.png
    ├── shap_waterfall.png
    ├── risk_tiers.png
    └── financial_exposure.png
```

---

## 🔬 Methodology

The project follows a rigorous, end-to-end machine learning workflow:

### 1. Exploratory Data Analysis
- Attrition rate and class distribution
- Numerical feature distributions by attrition status (Age, Income, Tenure)
- Attrition rates across categorical features (Department, Job Role, Overtime, Travel)
- Satisfaction score analysis (Job, Environment, Work-Life Balance, Involvement)
- Full feature correlation matrix

### 2. Feature Engineering
Ten domain-specific HR features were engineered from raw data:

| Feature | Description |
|---|---|
| `IncomePerYear` | Monthly income ÷ total working years — compensation relative to experience |
| `PromotionLag` | Years since last promotion minus years in current role |
| `LoyaltyScore` | Years at company ÷ number of companies worked — career mobility signal |
| `OverallSatisfaction` | Composite mean of all four satisfaction scores |
| `ManagerStability` | Years with current manager ÷ years at company |
| `IsOverTime` | Binary flag for overtime status |
| `IsFrequentTraveller` | Binary flag for frequent business travel |
| `IsSingle` | Binary flag for marital status |
| `LongTimeNoPromo` | Flag: no promotion in 5+ years |
| `LowIncome` | Flag: income in bottom 25th percentile |

### 3. Preprocessing & Class Imbalance
- Label encoding for categorical variables
- `StandardScaler` for numerical normalisation
- 70 / 15 / 15 stratified train / validation / test split
- **SMOTE** (Synthetic Minority Over-sampling Technique) applied to the training set only — preventing data leakage into validation and test sets

### 4. Model Training
Five algorithms trained and compared on the same data:

| Model | Type |
|---|---|
| Logistic Regression | Interpretable linear baseline |
| Decision Tree | Rule-based, explainable |
| Random Forest | Ensemble bagging |
| Gradient Boosting | Sequential ensemble |
| **XGBoost** ⭐ | High-performance gradient boosting (primary model) |

### 5. Hyperparameter Tuning
- `GridSearchCV` with `StratifiedKFold` (k=5) on XGBoost
- Optimised for AUC-ROC to balance precision and recall
- Parameters tuned: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`

### 6. Explainability (SHAP)
- Global SHAP beeswarm and bar plots — top 15 attrition drivers across the workforce
- Individual waterfall plot — explains each specific employee's risk score to HR managers

### 7. Risk Scoring & Financial Analysis
- All 1,470 employees scored with attrition probability
- Four-tier risk classification applied
- Replacement cost model: predicted probability × annual salary × 150% industry-standard replacement factor

---

## 🤖 Models & Results

### Validation Set Performance

| Model | AUC-ROC | F1-Score | Recall | Precision |
|---|---|---|---|---|
| **XGBoost (Tuned)** ⭐ | **~0.86** | **~0.72** | **~0.78** | **~0.67** |
| Random Forest | ~0.84 | ~0.68 | ~0.72 | ~0.65 |
| Gradient Boosting | ~0.83 | ~0.67 | ~0.71 | ~0.64 |
| Logistic Regression | ~0.78 | ~0.60 | ~0.65 | ~0.56 |
| Decision Tree | ~0.74 | ~0.62 | ~0.68 | ~0.57 |

> **Note:** In HR attrition, **Recall is the most critical metric** — a missed resignation (False Negative) is far more costly to the organisation than a false alarm (False Positive). XGBoost delivered the strongest overall performance across all relevant HR metrics.

### Risk Tier Distribution

| Tier | Threshold | Employees |
|---|---|---|
| 🔴 Critical Risk | ≥ 70% probability | ~5–8% of workforce |
| 🟠 High Risk | 50–69% probability | ~8–12% of workforce |
| 🟡 Medium Risk | 30–49% probability | ~15–20% of workforce |
| 🟢 Low Risk | < 30% probability | ~60–70% of workforce |

---

## 💡 Key Findings

Based on SHAP analysis, the top drivers of employee attrition are:

| Rank | Driver | Direction | Insight |
|---|---|---|---|
| 1 | **OverTime** | ↑ Risk | The single strongest predictor — employees working overtime are significantly more likely to resign |
| 2 | **MonthlyIncome** | ↓ Risk | Higher compensation strongly reduces attrition |
| 3 | **OverallSatisfaction** | ↓ Risk | Low satisfaction across job, environment, and work-life balance is a strong early warning signal |
| 4 | **YearsAtCompany** | ↓ Risk | Early-tenure employees (0–2 years) are the most vulnerable cohort |
| 5 | **PromotionLag** | ↑ Risk | Employees stuck in the same role without promotion become flight risks |
| 6 | **LoyaltyScore** | ↓ Risk | Employees with many prior employers are more likely to move again |
| 7 | **DistanceFromHome** | ↑ Risk | Long commutes meaningfully increase resignation probability |
| 8 | **IsSingle** | ↑ Risk | Single employees have fewer location constraints — higher mobility |
| 9 | **IsFrequentTraveller** | ↑ Risk | Frequent business travel correlates strongly with attrition |
| 10 | **StockOptionLevel** | ↓ Risk | Equity ownership is a powerful and cost-effective retention tool |

---

## 🏥 Actionable HR Recommendations

1. **Eliminate unnecessary overtime** — overtime is the single strongest attrition driver. Immediately audit workload distribution for all employees in the Critical and High Risk tiers.

2. **Benchmark and adjust compensation** — conduct a market salary review for employees in the bottom income quartile, with priority given to high-skill and high-risk roles.

3. **Launch a structured 30-60-90 day onboarding programme** — employees in their first two years are the most vulnerable. Structured onboarding has been shown to reduce early attrition by up to 50%.

4. **Fix promotion pipelines** — any employee in the same role for 3+ years without a promotion should be in an active career development conversation immediately.

5. **Expand stock option eligibility** — equity options are a high-impact, relatively low-cost retention lever. Extend eligibility to employees below manager level who are flagged at risk.

6. **Address commute burden** — offer remote or hybrid working arrangements to high-risk employees with commutes exceeding 20km.

7. **Prioritise Critical Risk employees this quarter** — the financial impact analysis shows that the cost of retaining Critical Risk employees is a fraction of their replacement cost.

---

## ⚙️ Installation

### Run in Google Colab (Recommended — No Setup Required)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GIVEN-CHINYAMA/Employee-Attrition-Prediction/blob/main/Employee_Attrition_Prediction_HR_Analytics.ipynb)

Click the badge above to open the notebook directly in Google Colab. All dependencies are installed automatically in the first cell.

---

### Run Locally

**Prerequisites:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/GIVEN-CHINYAMA/Employee-Attrition-Prediction.git
cd Employee-Attrition-Prediction

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install xgboost shap imbalanced-learn scikit-learn pandas numpy \
            matplotlib seaborn plotly

# 4. Launch Jupyter
jupyter notebook Employee_Attrition_Prediction_HR_Analytics.ipynb
```

---

## 🚀 Usage

1. Open the notebook in Google Colab or Jupyter.
2. Run all cells in order — the dataset loads automatically.
3. Review the EDA visualisations to understand the workforce.
4. Inspect the model comparison table and ROC/PR curves.
5. Use the SHAP plots to identify the top attrition drivers for your organisation.
6. Export `risk_df` (the employee risk scoring table) to prioritise HR interventions.
7. Reference the financial impact section to build your retention investment business case.

---

## 🛠️ Technologies

| Category | Library | Version |
|---|---|---|
| Data Manipulation | `pandas`, `numpy` | Latest |
| Machine Learning | `scikit-learn` | ≥ 1.3 |
| Gradient Boosting | `xgboost` | ≥ 2.0 |
| Class Imbalance | `imbalanced-learn` (SMOTE) | ≥ 0.11 |
| Explainability | `shap` | ≥ 0.44 |
| Visualisation | `matplotlib`, `seaborn`, `plotly` | Latest |
| Environment | Python 3.10+ · Jupyter / Google Colab | — |

---

## 📋 Project Highlights

- ✅ **5 ML models** trained and compared on the same stratified data splits
- ✅ **SMOTE** for class imbalance — applied to training set only (zero leakage)
- ✅ **GridSearchCV** with Stratified K-Fold hyperparameter tuning
- ✅ **SHAP** global and individual explainability — legally and ethically compliant
- ✅ **10 domain-specific** HR features engineered from raw data
- ✅ **4-tier risk scoring** system for all 1,470 employees
- ✅ **Financial impact model** — replacement cost exposure by department
- ✅ **15 publication-quality** visualisations throughout
- ✅ **Zero manual data download** — dataset loads automatically

---

## 👤 Author

**Given Chinyama**  
Data Scientist · Kwame Nkrumah University · Lusaka, Zambia

[![GitHub](https://img.shields.io/badge/GitHub-GIVEN--CHINYAMA-181717?logo=github)](https://github.com/GIVEN-CHINYAMA)
[![Email](https://img.shields.io/badge/Email-givenchinyama%40gmail.com-D14836?logo=gmail&logoColor=white)](mailto:givenchinyama@gmail.com)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <i>Built with ❤️ for HR professionals, data scientists, and people analytics practitioners.</i>
</p>
