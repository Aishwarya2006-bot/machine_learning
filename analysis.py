import pandas as pd


def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)


def get_dataset_info(df):

    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Missing Values": df.isnull().sum().sum(),
        "Duplicates": df.duplicated().sum()
    }


def get_summary(df):
    return df.describe()


def get_missing_values(df):
    return df.isnull().sum()
