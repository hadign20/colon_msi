import os
import pandas as pd
import numpy as np
from src.feature_selection.correlation import remove_collinear_features
from src.feature_selection.feature_selection import *
from src.model.model_building import evaluate_models, save_classification_results


#=========================================
# set paths
#=========================================
data_path = r'D:\postdoc\projects\colonMSI\data'
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
CORR_THRESH = 0.8

FEATURE_SELECTION = True
FEATURE_SELECTION_METHOD = 'composite' # 'mrmr', 'pvalue', 'auc', 'composite'
min_num_features = 1
max_num_features = 5

MODEL_BUILDING = True
TRAIN_TEST_SPLIT = False
TEST_SIZE = 0.3
CROSS_VALIDATION = True
CV_FOLDS = 5
HYPERPARAMETER_TUNING = True


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
        df = remove_collinear_features(df, CORR_THRESH)

    if FEATURE_SELECTION:
        print("\n===================================")
        print("Performing feature analysis")
        print("===================================")
        p_values_df = calculate_p_values(df, outcome_column, categorical_columns, exclude_columns)
        auc_values_df = calculate_auc_values(df, outcome_column, categorical_columns, exclude_columns)
        mrmr_df = MRMR_feature_count(df, outcome_column, categorical_columns, exclude_columns, max_num_features, CV_FOLDS)
        composite_df = calculate_feature_scores(p_values_df, auc_values_df, mrmr_df, results_dir)

        save_feature_analysis(p_values_df, auc_values_df, mrmr_df, composite_df, results_dir)


        for num_features in range(min_num_features, max_num_features+1):
            print("\n===================================")
            print(f"Selecting {num_features} significant features")
            print("===================================")

            selected_features = []
            if FEATURE_SELECTION_METHOD == 'mrmr':
                selected_features = mrmr_df['Feature'][:num_features].tolist()
                print(f"{num_features} features were selected by using MRMR method")
            elif FEATURE_SELECTION_METHOD == 'pvalue':
                selected_features = p_values_df['Feature'][:num_features].tolist()
                print(f"{num_features} features were selected by using pvalue method")
            elif FEATURE_SELECTION_METHOD == 'auc':
                selected_features = auc_values_df['Feature'][:num_features].tolist()
                print(f"{num_features} features were selected by using auc method")
            elif FEATURE_SELECTION_METHOD == 'composite':
                selected_features = composite_df['Feature'][:num_features].tolist()
                print(f"{num_features} features were selected by a composite of MRMR, AUC, and MRMR method")
            else:
                raise ValueError("FEATURE_SELECTION_METHOD is not correct. It should be 'mrmr', 'pvalue', 'auc', or 'composite'")


            df = df[exclude_columns + selected_features + [outcome_column]]

            # =========================================
            # Model building and evaluation
            # =========================================
            if MODEL_BUILDING:



                TRAIN_TEST_SPLIT = False
                TEST_SIZE = 0.3
                CROSS_VALIDATION = True
                CV_FOLDS = 5
                HYPERPARAMETER_TUNING = True


                print("\n===================================")
                print(f"Trainig and evaluating classification models for {num_features} features")
                print("===================================")
                X = df.loc[:, ~df.columns.isin(exclude_columns + [outcome_column])]
                y = df[outcome_column]
                classification_results = evaluate_models(X, y, method=evaluation_method, **eval_kwargs)

                classification_results_file = os.path.join(results_dir, 'model_evaluation_results.xlsx')
                save_classification_results(classification_results, classification_results_file, num_features, method=evaluation_method)





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
