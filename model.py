import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def train_classification_model(df, target):

    data = df.copy()

    for col in data.columns:

        if data[col].dtype == "object":

            data[col] = data[col].fillna("Unknown")

            encoder = LabelEncoder()

            data[col] = encoder.fit_transform(
                data[col].astype(str)
            )

        else:

            data[col] = data[col].fillna(
                data[col].mean()
            )

    X = data.drop(columns=[target])

    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance":
        model.feature_importances_

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    return (
        accuracy,
        report,
        matrix,
        importance
    )
