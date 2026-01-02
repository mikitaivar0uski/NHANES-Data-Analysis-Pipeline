from pathlib import Path

# ==============================================================================
# 1. PATHS
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. FILE MAPPING (NHANES 2017-2018)
# ==============================================================================
NHANES_MAP = {
    # --- Demographics ---
    "DEMO_J": [
        "SEQN",
        "RIAGENDR",
        "RIDAGEYR",
        "INDFMPIR",
        "DMDEDUC2",
        "DMDMARTL",
        "WTMEC2YR",
        "SDMVPSU",
        "SDMVSTRA",
        "RIDRETH3",
    ],
    # --- Questionnaire ---
    "DPQ_J": ["SEQN"]
    + [f"DPQ0{i}0" for i in range(1, 10)],  # Generates DPQ010...DPQ090
    "HSQ_J": ["SEQN", "HSD010"],
    "SMQ_J": ["SEQN", "SMQ020"],
    "ALQ_J": ["SEQN", "ALQ111"],
    "PAQ_J": ["SEQN", "PAQ650"],
    # --- Exams & Lab ---
    "BMX_J": ["SEQN", "BMXBMI"],
    "BPX_J": ["SEQN", "BPXSY1", "BPXDI1"],
    "BIOPRO_J": [
        "SEQN",
        "LBXSGL",
        "LBXSCH",
        "LBXSTR",
        "LBXSUA",
        "LBXSIR",
        "LBXSCR",
        "LBXSATSI",
    ],
    "CBC_J": ["SEQN", "LBXWBCSI", "LBXHGB"],
    "HSCRP_J": ["SEQN", "LBXHSCRP"],
    "PBCD_J": ["SEQN", "LBXBCD", "LBXBPB", "LBXTHG"],
    "ALB_CR_J": ["SEQN", "URDACT"],
    "VID_J": ["SEQN", "LBXVIDMS"],
    "DXX_J": ["SEQN", "DXDTOPF", "DXDTOLE", "DXXTRFAT"],
}

# ==============================================================================
# 3. RENAME MAPPING (Human Readable Names)
# ==============================================================================
# ==============================================================================
# 3. RENAME MAPPING (Human Readable Names)
# ==============================================================================
RENAME_MAP = {
    # --- Demographics ---
    "RIAGENDR": "Gender",  # Gender
    "RIDAGEYR": "Age",  # Age
    "RIDRETH3": "Race",
    "INDFMPIR": "Poverty_Ratio",  # Ratio of family income to poverty threshold
    "DMDEDUC2": "Education_Level",  # Education level (adults 20+)
    "DMDMARTL": "Marital_Status",  # Marital status
    "INDHHIN2": "Family_Income_Category",  # Annual household income category
    "SDMVPSU": "PSU",
    "SDMVSTRA": "Strata",
    "WTMEC2YR": "MEC_Weight",
    # --- Habits & History ---
    "HSD010": "General_Health_Cond",  # General health condition (1-5 scale)
    "SLQ050": "Trouble_Sleeping_Doc",  # Ever told by doctor about sleep problems?
    "SMQ020": "100_Cigs_Lifetime",  # Smoked at least 100 cigarettes in life?
    "ALQ111": "Alcohol_Tried",  # Ever had at least one drink of alcohol?
    "PAQ650": "Vigorous_Activity",  # Vigorous intensity physical activity
    # --- Body Measures ---
    "BMXBMI": "BMI",  # Body Mass Index (kg/m^2)
    "BMXWT": "Weight_kg",  # Weight in kilograms
    "BMXWAIST": "Waist_cm",  # Waist circumference in centimeters
    # --- Cardiovascular ---
    "BPXSY1": "BP_Systolic",  # Systolic blood pressure (mmHg)
    "BPXDI1": "BP_Diastolic",  # Diastolic blood pressure (mmHg)
    "LBXSCH": "Cholesterol_Total_mgdL",  # Total cholesterol (mg/dL)
    # --- Biochemistry ---
    "LBXSGL": "Glucose_mgdL",  # Serum glucose (mg/dL)
    "LBXSUA": "UricAcid_mgdL",  # Uric acid (mg/dL)
    "LBXSTR": "Triglycerides_mgdL",  # Triglycerides (mg/dL)
    "LBXSIR": "Iron_ugdL",  # Refrigerated serum iron (ug/dL)
    "LBXSCR": "Creatinine_mgdL",  # Creatinine (kidney function marker) (mg/dL)
    "LBXSATSI": "Transferrin_Sat_Pct",  # Transferrin saturation percentage
    # --- Inflammation & Immunity ---
    "LBXHSCRP": "CRP_mgL",  # C-reactive protein (inflammation marker) (mg/L)
    "LBXWBCSI": "WBC_1000cells",  # White blood cell count (1000 cells/uL)
    "LBXHGB": "Hemoglobin_g_dL",  # Hemoglobin (g/dL)
    # --- Heavy Metals ---
    "LBXBCD": "Cadmium_ugL",  # Blood cadmium (ug/L)
    "LBXBPB": "Lead_ugdL",  # Blood lead (ug/dL)
    "LBXTHG": "Mercury_Total_ugL",  # Total blood mercury (ug/L)
    # --- Kidney & Vitamins ---
    "URDACT": "Albumin_Creatinine_Ratio",  # Albumin/creatinine ratio (renal health)
    "LBXVIDMS": "VitaminD_nmolL",  # Vitamin D (nmol/L)
    "DXDTOPF": "Body_Fat_Pct",  # Процент жира
    "DXDTOLE": "Lean_Mass_g",  # Мышечная масса (г)
    "DXXTRFAT": "Trunk_Fat_g",
}

# ==============================================================================
# 4. ENCODING MAPPING (Categories -> Numbers)
# ==============================================================================
# This is crucial for Correlation Matrix.
# We convert "1=Yes, 2=No" into "1=Yes, 0=No" to make math work.

ENCODING_LOGIC = {
    "Gender": {1: 0, 2: 1},  # 0=Male, 1=Female
    "Vigorous_Activity": {1: 1, 2: 0},  # 1=Active, 0=Not Active
    # Smoking: 1=Every day, 2=Some days -> 1 (Smoker). 3=Not at all -> 0 (Non-smoker)
}

# List of columns that are strictly numerical (for outlier cleaning)
NUMERICAL_COLS = [
    "Age",
    "Poverty_Ratio",
    "Household_Size",
    "Sleep_Hours",
    "Alcohol_Drinks_Day",
    "Sedentary_Time_Min",
    "BMI",
    "Weight_kg",
    "Waist_cm",
    "BP_Systolic",
    "BP_Diastolic",
    "Glucose_mgdL",
    "HbA1c_Pct",
    "Cholesterol_Total_mgdL",
    "HDL_Cholesterol",
    "UricAcid_mgdL",
    "Triglycerides_mgdL",
    "Iron_ugdL",
    "Ferritin_ngmL",
    "Creatinine_mgdL",
    "CRP_mgL",
    "WBC_1000cells",
    "Hemoglobin_g_dL",
    "Platelets_1000cells",
    "Cadmium_ugL",
    "Lead_ugdL",
    "Mercury_Total_ugL",
    "Albumin_Creatinine_Ratio",
    "VitaminD_nmolL",
    "LBXSATSI",
]
