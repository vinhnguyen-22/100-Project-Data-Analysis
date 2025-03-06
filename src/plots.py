from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR


def plot_histogram(df:pd.DataFrame, column_name:str)->None:
    plt.figure(figsize=(5, 3))
    sns.histplot(df [column_name], kde=True)
    plt.title(f"Distribution of {column_name}")
    # calculate the mean and median values for the columns
    col_mean =df[column_name].mean()
    col_median = df[column_name]. median()
    # add vertical lines for mean and median
    plt.axvline(col_mean, color="red", linestyle="--", label="Mean")
    plt.axvline(col_median, color="green", linestyle="-", label="Median")
    plt.legend()
    plt.show()

def plot_boxplot(df:pd.DataFrame, col: str)->None:
    sns.boxplot(x=col, data=df)
    plt.title(f'{col} Boxplot')
    plt.show()

def plot_coefficients_with_confidence_intervals(reg_model):
    """
    Plot the coefficients with confidence intervals.
    Parameters:
    coef_df (pd.DataFrame): DataFrame containing the features, coefficients, and confidence intervals.
    """
    # Sắp xếp DataFrame theo giá trị tuyệt đối của hệ số
    features = reg_model.params.index
    coefficients = reg_model.params.values
    conf = reg_model.conf_int()

    # Tạo DataFrame từ các hệ số và khoảng tin cậy
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefficients,
        'CI Lower': conf[0],
        'CI Upper': conf[1]
    })

    coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=True).index)

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.errorbar(coef_df['Coefficient'], coef_df['Feature'], xerr=[coef_df['Coefficient'] - coef_df['CI Lower'], coef_df['CI Upper'] - coef_df['Coefficient']], fmt='o', color='navy', ecolor='gray', capsize=3)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title('Coefficient Plot with Confidence Intervals')
    plt.show()
    
def plot_parabol(df, x_axis, y_axis, x_point, y_point):
    plt.rcParams["figure.figsize"] = [12, 8]
    sns.set_theme(font_scale=1.5)
    plot = sns.lineplot(data=df, x=x_axis, y=y_axis)
    plot.axhline(y=y_point, color='r', linestyle='--')
    plot.axvline(x=x_point, color='r', linestyle='--')
    plot.set_title('Predicted Profits by Price')

    # Add data label for the price with the maximum profit
    plot.annotate(f'{x_axis}: ${x_point}\n{y_axis}: ${y_point}', xy=(x_point, y_point), xytext=(x_point-0.2, y_point - 50000))
    plt.show()
    
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    pass
    # -----------------------------------------


if __name__ == "__main__":
    app()
