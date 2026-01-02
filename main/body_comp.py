import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.weightstats import DescrStatsW


# --- 1. ВНУТРЕННЯЯ ФУНКЦИЯ (Helper) ---
# Она должна быть определена ДО того, как её вызовет главная функция
def calculate_muscle_index(df):
    """Internal helper: Calculates Muscle Mass Index (Height-adjusted)."""
    local_df = df.copy()

    # А. Восстанавливаем Рост (Height in meters)
    # Формула: BMI = Weight / Height^2  ->  Height^2 = Weight / BMI
    if "Height_m" not in local_df.columns:
        if "Weight_kg" in local_df.columns and "BMI" in local_df.columns:
            # Избегаем деления на 0
            local_df["Height_m2"] = local_df["Weight_kg"] / local_df["BMI"].replace(
                0, np.nan
            )
        else:
            return None  # Не хватает данных
    else:
        local_df["Height_m2"] = local_df["Height_m"] ** 2

    # Б. Считаем Индекс (SMI - Skeletal Muscle Index proxy)
    # Lean_Mass_g (мышцы в граммах) / 1000 = кг
    # SMI = кг / м^2
    if "Lean_Mass_g" in local_df.columns:
        local_df["Muscle_Index"] = (local_df["Lean_Mass_g"] / 1000) / local_df[
            "Height_m2"
        ]
        return local_df
    else:
        return None


# --- 2. ГЛАВНАЯ ФУНКЦИЯ (Export) ---
def analyze_body_composition_story(df: pd.DataFrame):
    """
    Generates the Body Composition Story (3 Acts).
    """

    # --- A. Подготовка данных ---
    # Вызываем помощника, который лежит В ЭТОМ ЖЕ ФАЙЛЕ выше
    temp_df = calculate_muscle_index(df)

    if temp_df is not None:
        local_df = temp_df
        use_muscle_index = True
        muscle_col = "Muscle_Index"
        muscle_label = "Muscle Index (Height-Adjusted)"
    else:
        local_df = df.copy()
        use_muscle_index = False
        muscle_col = "Lean_Mass_g"
        muscle_label = "Absolute Muscle Mass (Weight-Biased!)"
        print("⚠️ Warning: Could not calculate Muscle Index. Using absolute mass.")

    # Таргет и Пол
    local_df["Dep_Numeric"] = pd.to_numeric(
        local_df["Depression"], errors="coerce"
    ).fillna(0)

    # Маппинг пола (проверка на тип данных)
    if pd.api.types.is_numeric_dtype(local_df["Gender"]):
        gender_map = {0: "Men", 1: "Women"}
    else:
        gender_map = {"0": "Men", "1": "Women"}
    local_df["Gender_Label"] = local_df["Gender"].map(gender_map)

    # --- B. Категоризация (Quintiles) ---

    # 1. BMI Detailed
    local_df["BMI_Detailed"] = pd.cut(
        local_df["BMI"],
        bins=[0, 18.5, 25, 30, 35, 40, 300],
        labels=[
            "Underweight\n(<18.5)",
            "Normal\n(18.5-25)",
            "Overweight\n(25-30)",
            "Obese I\n(30-35)",
            "Obese II\n(35-40)",
            "Morbid\n(>40)",
        ],
    )

    # 2. Fat Quintiles
    if "Body_Fat_Pct" in local_df.columns:
        local_df["Fat_Quintile"] = local_df.groupby("Gender_Label")[
            "Body_Fat_Pct"
        ].transform(
            lambda x: pd.qcut(
                x, 5, labels=["Q1 (Leanest)", "Q2", "Q3", "Q4", "Q5 (Fattest)"]
            )
        )

    # 3. Muscle Quintiles
    if muscle_col in local_df.columns:
        local_df["Muscle_Quintile"] = local_df.groupby("Gender_Label")[
            muscle_col
        ].transform(
            lambda x: pd.qcut(
                x, 5, labels=["Q1 (Low Muscle)", "Q2", "Q3", "Q4", "Q5 (High Muscle)"]
            )
        )

    # --- C. Функция расчета статистики ---
    def get_stats(col):
        stats = []
        if col not in local_df.columns:
            return pd.DataFrame()

        for gen in ["Men", "Women"]:
            sub = local_df[local_df["Gender_Label"] == gen]
            if sub.empty:
                continue

            for cat in sub[col].dropna().unique().sort_values():
                grp = sub[sub[col] == cat]
                if len(grp) > 30:
                    wm = DescrStatsW(grp["Dep_Numeric"], weights=grp["MEC_Weight"]).mean
                    stats.append({"Category": cat, "Gender": gen, "Rate_Pct": wm * 100})
        return pd.DataFrame(stats)

    # --- D. Визуализация (3 Акта) ---
    sns.set_theme(style="white", context="talk", font_scale=0.9)

    # ACT 1: BMI
    plt.figure(figsize=(16, 6))
    df_bmi = get_stats("BMI_Detailed")
    if not df_bmi.empty:
        ax = sns.barplot(
            x="Category",
            y="Rate_Pct",
            hue="Gender",
            data=df_bmi,
            palette={"Men": "#3498db", "Women": "#e74c3c"},
            edgecolor="black",
        )
        plt.title("ACT 1: The BMI Illusion", fontweight="bold")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.grid(axis="y", alpha=0.3)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.1f%%", padding=3, fontsize=10)
        plt.show()

    # ACT 2: FAT
    plt.figure(figsize=(16, 6))
    df_fat = get_stats("Fat_Quintile")
    if not df_fat.empty:
        ax = sns.barplot(
            x="Category",
            y="Rate_Pct",
            hue="Gender",
            data=df_fat,
            palette={"Men": "#2980b9", "Women": "#c0392b"},
            edgecolor="black",
        )
        plt.title("ACT 2: The Truth (Body Fat %)", fontweight="bold")
        plt.xlabel("Fat Quintiles")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.grid(axis="y", alpha=0.3)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.1f%%", padding=3, fontsize=10)
        plt.show()

    # ACT 3: MUSCLE
    plt.figure(figsize=(16, 6))
    df_muscle = get_stats("Muscle_Quintile")
    if not df_muscle.empty:
        # Men = Green (Health), Women = Gold
        ax = sns.barplot(
            x="Category",
            y="Rate_Pct",
            hue="Gender",
            data=df_muscle,
            palette={"Men": "#2ecc71", "Women": "#f1c40f"},
            edgecolor="black",
        )
        plt.title(f"ACT 3: The Muscle Armor ({muscle_label})", fontweight="bold")
        plt.xlabel("Muscle Index Quintiles (Corrected for Height)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.grid(axis="y", alpha=0.3)

        # Добавляем пояснение
        plt.figtext(
            0.5,
            -0.05,
            "Trend: If Q1 (Low Muscle) is high -> Sarcopenia Risk.",
            ha="center",
            fontsize=11,
            bbox={"facecolor": "white", "alpha": 0.5},
        )

        for c in ax.containers:
            ax.bar_label(c, fmt="%.1f%%", padding=3, fontsize=10)
        plt.show()
