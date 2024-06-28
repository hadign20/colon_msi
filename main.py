import os
import pandas as pd
import numpy as np
from src.feature_selection.correlation import remove_collinear_features

#=========================================
# set paths
#=========================================
data_path = r'D:\projects\colonMSI\data'
result_dir = r'./results'
img_data_path = os.path.join(data_path, "Segmentations")
excel_file_name = "TumorTexture"
selected_sheet = "2_1"

features_file = os.path.join(data_path, excel_file_name + ".xlsx")
results_path = os.path.join(result_dir, excel_file_name)
os.makedirs(results_path, exist_ok=True)

#=========================================
# set parameters
#=========================================
FEATURE_CORRELATION = True
corr_thresh = 0.8


#=========================================



def main():
    df = pd.read_excel(features_file, sheet_name=selected_sheet)


    # =========================================
    # Feature selection
    # =========================================
    if FEATURE_CORRELATION:
        df = remove_collinear_features(df, corr_thresh)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
