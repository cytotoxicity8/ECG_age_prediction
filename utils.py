import numpy as np
import pandas as pd

def flattening_prediction(pred_age, idx):
    return pd.Series(pred_age.detach().cpu().flatten(), index=idx)

def validate_child(child_dataset, prediction_df):

    impute_dict = {"MALE":1.25, "FEMALE":1.33, "unknown":2.83}
    zero = np.zeros((12, 5000))
    for idx in range(len(child_dataset)):
        if np.any(np.all(child_dataset[idx][0] == zero, axis=1)):
            print(child_dataset.index_list[idx], "includes a channel with only zeros.")
            gender = prediction_df.loc[child_dataset.index_list[idx], "GENDER"]
            prediction_df.loc[child_dataset.index_list[idx], "AGE"] = impute_dict[gender]

    prediction_df.loc[prediction_df["AGE"]< 1/12, "AGE"] = 1/12

    return prediction_df


def validate_adult(adult_dataset, prediction_df):
    impute_dict = {"MALE":65, "FEMALE":63, "unknown":64}

    zero = np.zeros((12, 5000))

    for idx in range(len(adult_dataset)):
        if np.any(np.all(adult_dataset[idx][0] == zero, axis=1)):
            print(adult_dataset.index_list[idx], "includes a channel with only zeros.")
            gender = prediction_df.loc[adult_dataset.index_list[idx], "GENDER"]
            prediction_df.loc[adult_dataset.index_list[idx], "AGE"] = impute_dict[gender]

    prediction_df.loc[prediction_df["AGE"] > 105, "AGE"] = 105

    return prediction_df