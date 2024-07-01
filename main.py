import os
import pandas as pd
import numpy as np
from src.feature_selection.correlation import remove_collinear_features
from src.feature_selection.feature_analysis import *

#=========================================
# set paths
#=========================================
data_path = r'D:\projects\colonMSI\data'
result_path = r'./results'
img_data_path = os.path.join(data_path, "Segmentations")
excel_file_name = "TumorTexture"
selected_sheet = "2_1"
outcome_column = "Outcome"
exclude_columns = ["Case"]
categorical_columns = []

features_file = os.path.join(data_path, excel_file_name + ".xlsx")
results_dir = os.path.join(result_path, excel_file_name)
os.makedirs(results_dir, exist_ok=True)

#=========================================
# set parameters
#=========================================
FEATURE_CORRELATION = True
FEATURE_ANALYSIS = True
FEATURE_SELECTION = True
corr_thresh = 0.8


#=========================================



def main():
    df = pd.read_excel(features_file, sheet_name=selected_sheet)


    # =========================================
    # Feature selection
    # =========================================
    if FEATURE_CORRELATION:
        df = remove_collinear_features(df, corr_thresh)

    if FEATURE_ANALYSIS:
        p_values_df = calculate_p_values(df, outcome_column, categorical_columns, exclude_columns)
        auc_values_df = calculate_auc_values(df, outcome_column, categorical_columns, exclude_columns)
        save_feature_analysis(p_values_df, auc_values_df, results_dir)

    if FEATURE_SELECTION:





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
