import pandas as pd
from sklearn.datasets import load_iris
import os


def get_data():
    # load iris dataset
    iris = load_iris()

    # convert to dataframe
    df = pd.DataFrame(
        data=iris.data,
        columns=iris.feature_names
    )

    # add target column
    df['target'] = iris.target
    df['target_name'] = df['target'].map({
        0: 'setosa',
        1: 'versicolor',
        2: 'virginica'
    })

    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nTarget distribution:")
    print(df['target_name'].value_counts())

    # save to data folder
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/iris.csv', index=False)
    print(f"\n✅ Data saved to data/iris.csv")

    return df


if __name__ == "__main__":
    get_data()