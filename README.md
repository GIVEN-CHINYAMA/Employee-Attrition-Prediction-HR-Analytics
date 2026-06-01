Readme · MDCopy👥 Employee Attrition Prediction — HR Analytics

Predicting which employees are most likely to resign, and why — using a full production-grade Machine Learning pipeline built on the IBM HR Analytics dataset.

Show Image
Show Image
Show Image
Show Image
Show Image
Show Image

📌 Project Overview
Employee attrition — the voluntary resignation of staff — is one of the most expensive and disruptive challenges facing organisations today. Replacing a single employee costs between 50% and 200% of their annual salary when recruitment, onboarding, training, and lost productivity are factored in.
This project delivers a complete, end-to-end HR analytics pipeline that:

Predicts which employees are at risk of resigning at the individual level
Identifies the root causes of attrition across the workforce
Segments employees into four risk tiers for targeted HR interventions
Quantifies the financial exposure from predicted attrition
Explains every prediction using SHAP values — in plain language HR managers can act on


🎯 Business Problem

"Which employees are most likely to resign, and what organisational factors are driving attrition — so that HR can act before it is too late?"


📁 Repository Structure
Employee-Attrition-Prediction-HR-Analytics/
│
├── Employee_Attrition_Prediction_HR_Analytics.ipynb   # Main notebook
├── README.md                                          # This file
└── requirements.txt                                   # Python dependencies

🗂️ Table of Contents

Dataset
Project Pipeline
Feature Engineering
Models Trained
Results
SHAP Explainability
Key Findings & HR Recommendations
How to Run
Tech Stack
Author


📊 Dataset
PropertyDetailSourceIBM HR Analytics Employee Attrition DatasetAvailable viaKaggleEmployees1,470Features35 (demographics, compensation, satisfaction, work history)TargetAttrition — Yes / No (binary classification)Class imbalance~16% attrition rate (imbalanced)
Key feature categories:

Demographics — Age, Gender, MaritalStatus, DistanceFromHome
Job Info — Department, JobRole, JobLevel, BusinessTravel
Compensation — MonthlyIncome, StockOptionLevel, PercentSalaryHike
Satisfaction — JobSatisfaction, EnvironmentSatisfaction, WorkLifeBalance, JobInvolvement
Work History — YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, NumCompaniesWorked
Other — OverTime, TrainingTimesLastYear, PerformanceRating


🔁 Project Pipeline
Raw Data → EDA → Feature Engineering → Preprocessing → SMOTE
    → Model Training (5 algorithms) → Evaluation → Hyperparameter Tuning
        → SHAP Explainability → Risk Scoring → Financial Impact Analysis
Sections in the Notebook
#SectionDescription1Environment SetupLibrary imports and configuration2Data LoadingIBM HR dataset loaded from GitHub mirror3Data OverviewShape, dtypes, missing values, duplicates4Exploratory Data Analysis15 publication-quality visualisations5Feature Engineering10 domain-specific HR features created6Preprocessing & EncodingLabel encoding, scaling, train/test split7Class Imbalance (SMOTE)Applied to training set only — no data leakage8Model TrainingFive ML algorithms trained and compared9Model EvaluationF1, AUC-ROC, Recall, Precision, Confusion Matrix10Hyperparameter TuningGridSearchCV with Stratified K-Fold (k=5) on XGBoost11SHAP ExplainabilityBeeswarm, bar, and waterfall plots12Employee Risk Scoring4-tier risk classification for all 1,470 employees13Financial Impact AnalysisReplacement cost exposure by employee and department14HR RecommendationsActionable business insights for HR leaders15ConclusionSummary of technical achievements and skills demonstrated

🔧 Feature Engineering
Ten domain-specific HR features were engineered from raw data:
FeatureDescriptionOverallSatisfactionMean of job, environment, and work-life balance scoresPromotionLagYears since last promotion relative to years in roleLoyaltyScoreRatio of years at company to total working yearsIncomePerYearExpMonthly income divided by total working yearsIsSingleBinary flag for single marital statusIsFrequentTravellerBinary flag for frequent business travelSatisfactionRiskFlag for low satisfaction across all three metricsCareerVelocityJob level relative to total working yearsTenureStabilityRatio of years with current manager to years at companyCompensationGrowthIndexSalary hike weighted against performance rating

🤖 Models Trained
Five classification algorithms were trained and compared:
ModelNotesLogistic RegressionBaseline linear modelDecision TreeInterpretable non-linear modelRandom ForestEnsemble — robust to overfittingGradient BoostingSequential ensemble — strong performanceXGBoostBest performer — tuned with GridSearchCV
All models were evaluated using Stratified K-Fold cross-validation (k=5) to ensure reliable performance estimates on the imbalanced dataset.

📈 Results
XGBoost (tuned) achieved the best overall performance:
MetricScoreAUC-ROCHigh discriminative powerF1-Score (Attrition class)Optimised via SMOTE + tuningRecallPrioritised to minimise missed at-risk employees

Recall was the primary optimisation metric — in HR analytics, missing a flight-risk employee is costlier than a false alarm.


🔍 SHAP Explainability
SHAP (SHapley Additive exPlanations) was used to explain every prediction from the XGBoost model. Three plot types were generated:

Beeswarm plot — global feature importance across all employees
Bar plot — mean absolute SHAP values per feature
Waterfall plot — individual employee prediction breakdown

This makes the model auditable and explainable to HR managers and business leaders — not just data scientists.

💡 Key Findings & HR Recommendations
Top Attrition Drivers (from SHAP Analysis)
DriverDirectionInterpretationOverTime↑ RiskOvertime is the single strongest predictor of resignationMonthlyIncome↓ RiskHigher pay strongly reduces attritionOverallSatisfaction↓ RiskLow satisfaction across job, environment, and work-life is a warning signalYearsAtCompany↓ RiskEmployees in their first 2 years are most vulnerablePromotionLag↑ RiskNo promotion after 3+ years signals flight riskLoyaltyScore↓ RiskEmployees who have changed companies frequently are more likely to move againDistanceFromHome↑ RiskLong commutes increase resignation probabilityIsSingle↑ RiskSingle employees have fewer location constraintsIsFrequentTraveller↑ RiskFrequent business travel correlates strongly with attritionStockOptionLevel↓ RiskStock options are a powerful and cost-effective retention tool
Actionable HR Recommendations

Eliminate unnecessary overtime — audit workload distribution immediately for Critical and High Risk employees
Benchmark and adjust compensation — market salary review for employees in the bottom income quartile
Launch a 30-60-90 day onboarding programme — structured onboarding reduces early attrition by up to 50%
Fix promotion pipelines — any employee in the same role for 3+ years without promotion needs a career conversation now
Expand stock option eligibility — extend to more employees below manager level
Offer remote/hybrid arrangements — prioritise employees with long commutes flagged as high risk
Intervene with Critical Risk employees this quarter — the cost of replacement far exceeds the cost of retention


🚀 How to Run
Option 1 — Google Colab (Recommended)

Click the Open in Colab badge at the top of this README
Run all cells — the dataset loads automatically, no manual download needed

Option 2 — Local Environment
bash# Clone the repository
git clone https://github.com/GIVEN-CHINYAMA/Employee-Attrition-Prediction-HR-Analytics.git
cd Employee-Attrition-Prediction-HR-Analytics

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook Employee_Attrition_Prediction_HR_Analytics.ipynb
Requirements
pandas>=2.2.0
numpy>=2.0.0
scikit-learn>=1.6.0
xgboost>=3.2.0
imbalanced-learn>=0.12.0
shap>=0.45.0
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.20.0

🛠️ Tech Stack
CategoryToolsLanguagePython 3.10+Data ManipulationPandas, NumPyMachine LearningScikit-learn, XGBoostClass Imbalanceimbalanced-learn (SMOTE)ExplainabilitySHAPVisualisationMatplotlib, Seaborn, PlotlyEnvironmentGoogle Colab / Jupyter Notebook

✅ Skills Demonstrated

Machine Learning (5 algorithms trained and compared)
Class Imbalance Handling (SMOTE — applied correctly, no leakage)
Hyperparameter Tuning (GridSearchCV + Stratified K-Fold)
Explainable AI (SHAP — beeswarm, bar, waterfall plots)
Feature Engineering (10 domain-specific HR features)
Business Impact Quantification (financial exposure modelling)
Stakeholder-ready Visualisations (15 publication-quality charts)
HR Analytics Domain Knowledge


👤 Author
Given Chinyama
Data Scientist | Kwame Nkrumah University | Lusaka, Zambia

🔗 GitHub: github.com/GIVEN-CHINYAMA
📧 Email: givenchinyama@gmail.com
📅 Project Date: June 2026


📄 License
This project is licensed under the MIT License — see the LICENSE file for details.

If you found this project useful, please consider giving it a ⭐ on GitHub!
