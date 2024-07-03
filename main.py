import os
import pandas as pd
import numpy as np
from src.feature_selection.correlation import remove_collinear_features
from src.feature_selection.feature_selection import *
from src.model.model_building import evaluate_models, save_classification_results

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
MODEL_BUILDING = True
corr_thresh = 0.8
num_features = 20
num_CV_folds = 20
evaluation_method = 'cross_validation' # 'train_test_split' or 'cross_validation'
eval_kwargs = {'test_size': 0.2, 'random_state': 42} if evaluation_method == 'train_test_split' else {'cv': 5}


#=========================================



def main():
    df = pd.read_excel(features_file, sheet_name=selected_sheet)


    # =========================================
    # Feature selection
    # =========================================
    if FEATURE_CORRELATION:
        print("\n===================================")
        print("Removing correlated features")
        print("===================================")
        df = remove_collinear_features(df, corr_thresh)

    if FEATURE_SELECTION:
        print("\n===================================")
        print("Selecting significant features")
        print("===================================")
        p_values_df = calculate_p_values(df, outcome_column, categorical_columns, exclude_columns)
        auc_values_df = calculate_auc_values(df, outcome_column, categorical_columns, exclude_columns)
        mrmr_df = MRMR_feature_count(df, outcome_column, categorical_columns, exclude_columns, num_features, num_CV_folds)
        save_feature_analysis(p_values_df, auc_values_df, mrmr_df, results_dir)

        selected_features = mrmr_df['Feature'][:num_features].tolist()
        df = df[exclude_columns + selected_features + [outcome_column]]

    # =========================================
    # Model building and evaluation
    # =========================================
    if MODEL_BUILDING:
        print("\n===================================")
        print("Trainig and evaluating classification models")
        print("===================================")
        X = df.loc[:, ~df.columns.isin(exclude_columns + [outcome_column])]
        y = df[outcome_column]
        classification_results = evaluate_models(X, y, method=evaluation_method, **eval_kwargs)

        classification_results_file = os.path.join(results_dir, 'model_evaluation_results.xlsx')
        save_classification_results(classification_results, classification_results_file, method=evaluation_method)





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
