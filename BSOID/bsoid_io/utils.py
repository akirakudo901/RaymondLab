# Author: Akira Kudo
# Created: 2024/04/02
# Last Updated: 2024/04/03

import pandas as pd

def read_BSOID_labeled_csv(csv_path : str,
                           include_scorer : bool=False):
    """
    Reads a BSOID-labeled csv file to extract labels as well
    as the corresponding DLC data.

    :param str csv_path: Path to the BSOID labeled csv.
    :param bool include_scorer: Whether to include 'scorer' row as header.
    :returns np.ndarray label: Label extracted from the csv.
    :returns pd.DataFrame dlc_data: DLC data extracted from the csv.
    """
    header_rows = [0,1,2] if include_scorer else [1,2]
    df = pd.read_csv(csv_path, header=header_rows, index_col=2)
    label = df.iloc[:, 1].to_numpy()
    dlc_data = df.iloc[:, 2:]
    return label, dlc_data

if __name__ == "__main__":
    import os
    FILE_OF_INTEREST = r"312152_m2DLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000.csv"
    LABELED_PREFIX = r"Mar-10-2023labels_pose_40Hz"

    MOUSETYPE_FOLDER = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\Leland B-SOID YAC128 Analysis" + \
                    r"\YAC128\YAC128" # r"\Q175\WT"

    LABELED_CSV_PATH = os.path.join(MOUSETYPE_FOLDER,  
                                    "CSV files", "BSOID", #r"csv/BSOID/Feb-27-2024"
                                    LABELED_PREFIX + FILE_OF_INTEREST)

    label, dlc_data = read_BSOID_labeled_csv(LABELED_CSV_PATH)
    print(f"label: {label}")
    print(f"label.shape: {label.shape}")
    print(f"dlc_data: {dlc_data}")
    print(f"dlc_data.columns: {dlc_data.columns}")