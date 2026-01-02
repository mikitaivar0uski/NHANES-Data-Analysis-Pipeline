import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def analyze_mercury_paradox(df: pd.DataFrame):
    """
    Visualizes the 'Mercury Paradox':
    1. Shows that depressed individuals have LOWER mercury levels (Boxplot).
    2. Explains this via Income: Wealthier people eat more fish, accumulating mercury (Scatterplot).

    Args:
        df (pd.DataFrame): DataFrame containing 'Mercury_Total_ugL', 'Depression', 'Poverty_Ratio'.
    """

    # --- 1. Data Validation and Preparation ---
    required_cols = ["Mercury_Total_ugL", "Depression", "Poverty_Ratio"]
    if not set(required_cols).issubset(df.columns):
        print(
            f"⚠️ Error: Missing columns in DataFrame: {set(required_cols) - set(df.columns)}"
        )
        return

    # Work on a copy
    local_df = df.copy()

    # Log-transformation
    local_df["Log_Mercury"] = np.log10(local_df["Mercury_Total_ugL"] + 0.01)

    # Convert Depression target to string labels
    local_df["Depression_Label"] = (
        local_df["Depression"]
        .astype(str)
        .replace({"0": "No (0)", "0.0": "No (0)", "1": "Yes (1)", "1.0": "Yes (1)"})
    )

    local_df = local_df[local_df["Depression_Label"].isin(["No (0)", "Yes (1)"])]

    # --- 2. Visualization Setup ---
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Palette
    custom_palette = {"No (0)": "#2ecc71", "Yes (1)": "#e74c3c"}

    # --- 3. Plot A: The Paradox (Boxplot) ---
    # FIX: Added hue='Depression_Label' and legend=False to fix Seaborn warning
    sns.boxplot(
        x="Depression_Label",
        y="Log_Mercury",
        data=local_df,
        ax=axes[0],
        palette=custom_palette,
        hue="Depression_Label",  # <--- FIX 1
        legend=False,  # <--- FIX 2
        showfliers=False,
        order=["No (0)", "Yes (1)"],
    )

    # FIX: Removed emojis to fix Font warning
    axes[0].set_title(
        "The Paradox: Lower Mercury in Depressed Group", fontsize=14, fontweight="bold"
    )
    axes[0].set_xlabel("Depression Status")
    axes[0].set_ylabel("Log10 Mercury (ug/L)")

    # --- 4. Plot B: The Explanation (Regression Plot) ---
    sample_size = min(2000, len(local_df))
    sample = local_df.sample(sample_size, random_state=42)

    sns.regplot(
        x="Poverty_Ratio",
        y="Log_Mercury",
        data=sample,
        ax=axes[1],
        scatter_kws={"alpha": 0.3, "color": "grey", "s": 15},
        line_kws={"color": "#e74c3c", "linewidth": 2},
    )

    # FIX: Removed emojis
    axes[1].set_title(
        'The Explanation: "Wealthier People Eat Fish"', fontsize=14, fontweight="bold"
    )
    axes[1].set_xlabel("Poverty Income Ratio (PIR)\n(> 5.0 indicates high income)")
    axes[1].set_ylabel("Log10 Mercury")

    axes[1].text(
        x=0.5,
        y=sample["Log_Mercury"].max() * 0.9,
        s="Mercury rises with Income\n(Proxy for Seafood Consumption)",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="red"),
    )

    plt.tight_layout()
    plt.show()
