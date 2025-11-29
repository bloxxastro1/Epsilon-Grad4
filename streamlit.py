import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc
)

from sklearn.decomposition import PCA


# -------------------------------
# Streamlit App
# -------------------------------

st.title("🐧 Penguin Classification ML App")
st.write("A full ML pipeline with preprocessing, training, tuning, and evaluation.")

github_url = "https://github.com/bloxxastro1/Epsilon-Grad4/blob/main/penguins_size.csv"
df = pd.read_csv(github_url, sep=",", engine="python", on_bad_lines="skip")
st.success("✅ Data loaded successfully!")
st.write(f"Dataset shape: {df.shape}")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # Feature Engineering
    if "culmen_length_mm" in df.columns and "culmen_depth_mm" in df.columns:
        df["culmen_area"] = df["culmen_length_mm"] * df["culmen_depth_mm"]

    if "body_mass_g" in df.columns:
        df["body_mass_per_flipper"] = df["body_mass_g"] / df["flipper_length_mm"]

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Input / Output split
    y = df["species"]
    X = df.drop(columns=["species"])

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Identify columns
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", MinMaxScaler(), num_cols)
        ]
    )

    # Select model
    model_choice = st.selectbox(
        "Choose model",
        ["SVM", "Random Forest", "Logistic Regression", "KNN"]
    )

    if model_choice == "SVM":
        classifier = SVC(probability=True)
        param_grid = {
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["rbf", "poly", "sigmoid"],
            "classifier__gamma": ["scale", 0.01, 0.001],
        }

    elif model_choice == "Random Forest":
        classifier = RandomForestClassifier()
        param_grid = {
            "classifier__n_estimators": [100, 300],
            "classifier__max_depth": [5, 10, None]
        }

    elif model_choice == "Logistic Regression":
        classifier = LogisticRegression(max_iter=500)
        param_grid = {
            "classifier__C": [0.1, 1, 10]
        }

    else:
        classifier = KNeighborsClassifier()
        param_grid = {
            "classifier__n_neighbors": [3, 5, 7]
        }

    # Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    # Grid Search
    st.write("### Training & Hyperparameter Tuning...")
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    st.success("Training Done!")
    st.write("### Best Parameters", grid.best_params_)
    st.write("### Best CV Score", grid.best_score_)

    # Evaluation
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    st.write("### Test Accuracy", best_model.score(X_test, y_test))
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion Matrix
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=le.classes_).plot(ax=ax)
    st.pyplot(fig)

    # ROC Curve
    st.write("### ROC Curve (Multiclass)")
    fig, ax = plt.subplots()

    y_test_bin = label_binarize(y_test, classes=np.arange(len(le.classes_)))
    y_score = best_model.predict_proba(X_test)

    for i, class_name in enumerate(le.classes_):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, label=f"{class_name}")

    ax.plot([0, 1], [0, 1], "--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    st.pyplot(fig)

    # PCA Visualization
    st.write("### PCA Visualization (2D)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(preprocessor.fit_transform(X))

    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded)
    plt.legend(handles=scatter.legend_elements()[0], labels=le.classes_)
    st.pyplot(fig)

    st.success("Done!")

