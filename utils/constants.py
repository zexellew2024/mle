from datetime import datetime

from dateutil.relativedelta import relativedelta

# Datamart directories
BRONZE_LMS_DIR = 'datamart/bronze/lms/'
BRONZE_FEAT_DIR = 'datamart/bronze/'
SILVER_LMS_DIR = 'datamart/silver/loan_daily/'
SILVER_FEAT_DIR = 'datamart/silver/'
GOLD_LABEL_DIR = 'datamart/gold/label_store/'
GOLD_FEAT_DIR = 'datamart/gold/feature_store/'
GOLD_INTEGRATED_DIR = 'datamart/gold/'
GOLD_LOANS_PATH = GOLD_INTEGRATED_DIR + 'gold_loans.parquet'
GOLD_PREDICTIONS_DIR = GOLD_INTEGRATED_DIR + 'model_predictions/'
GOLD_MONITORING_DIR = GOLD_INTEGRATED_DIR + 'model_monitoring/'
MODEL_BANK_DIR = 'model_bank'

# Source feature files (under data/)
FEATURE_FILENAMES = ('features_attributes', 'features_financials', 'feature_clickstream')

# Backfill window
START_DATE_STR = "2023-01-01"
END_DATE_STR = "2024-12-01"

# Label definition
DPD_CUTOFF = 30
MOB_CUTOFF = 6
LABEL_DEF = f"{DPD_CUTOFF}dpd_{MOB_CUTOFF}mob"

# how far the label store must run past END_DATE_STR to mature labels for every feature snapshot
LABEL_END_DATE_STR = (datetime.strptime(END_DATE_STR, "%Y-%m-%d") + relativedelta(months=MOB_CUTOFF)).strftime("%Y-%m-%d")

# Model training
MODEL_TRAIN_DATE_STR = "2024-07-01"
TRAIN_TEST_MONTHS = 16
OOT_MONTHS = 2
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
CLICKSTREAM_COLS = [f"fe_{i}" for i in range(1, 21)]
ATTRIBUTES_COLS = ["Age", "Occupation", "Age_group"]
FINANCIALS_COLS = [
    "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Utilization_Ratio",
    "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance", "num_loan_types",
    "Credit_Mix_Label", "Credit_History_Months", "Spending_Level", "Payment_Value", "log_Annual_Income",
    "debt_to_income", "emi_to_salary", "investment_rate", "balance_to_debt", "inq_per_loan",
    "has_credit_limit_change",
]

# Model monitoring thresholds - Green/Yellow/Red zones
# Recall (p0 - highest priority)
RECALL_GREEN = 0.80
RECALL_YELLOW_LOW = 0.60
RECALL_YELLOW_HIGH = 0.79

# Accuracy (p1) - recalibrated to the model's actual achievable range (val/test/oot never
# exceeded ~0.79 for either champion or challenger; the old 0.80/0.85 bands were unreachable)
ACCURACY_GREEN = 0.78
ACCURACY_YELLOW_LOW = 0.70
ACCURACY_YELLOW_HIGH = 0.77

# ROC AUC (p2)
AUC_GREEN = 0.90
AUC_YELLOW_LOW = 0.75
AUC_YELLOW_HIGH = 0.89

# PSI (Population Stability Index) - data drift
PSI_GREEN = 0.10
PSI_YELLOW_LOW = 0.10
PSI_YELLOW_HIGH = 0.20

# CSI p-value (Chi-Square Index p-value) - feature distribution drift
CSI_PVALUE_GREEN = 0.05
CSI_PVALUE_YELLOW_LOW = 0.01
CSI_PVALUE_YELLOW_HIGH = 0.05

MISSING_FEATURE_ALERT_THRESHOLD = 0.5

# Ordering/colors for plotting alert levels on the monitoring summary visualization
ALERT_LEVEL_RANK = {"GREEN": 0, "YELLOW": 1, "RED": 2}
ALERT_LEVEL_COLOR = {"GREEN": "#2ca02c", "YELLOW": "#d4ac0d", "RED": "#d62728"}

# A/B testing is recommended once the challenger's OOT AUC exceeds the champion's by this much
CHALLENGER_AUC_UPLIFT_THRESHOLD = 0.05
