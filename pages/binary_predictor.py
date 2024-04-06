import streamlit as st
import pandas as pd
import os
# import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score 

userDir = os.path.dirname(__file__)


@st.cache_data(persist=True)
def load_data():
    data = pd.read_csv(userDir + '/portfolio/pages/mushrooms.csv')
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data


@st.cache_data(persist=True)
def split(df):
    y = df.type  # answer
    x = df.drop(columns=['type'])  # drop answer column
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test


def model_results(df, model, metrics):
    class_names = ['edible', 'poisonous']
    x_train, x_test, y_train, y_test = split(df)
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    y_pred = model.predict(x_test)

    col1, col2, col3 = st.columns(3)
    col1.write(f"Accuracy: {accuracy:.2f}")
    col2.write(f"Precision: {precision_score(y_test, y_pred, labels=class_names):.2f}")
    col3.write(f"Recall: {recall_score(y_test, y_pred, labels=class_names):.2f}")
    st.divider()

    plot_metrics(metrics, model, x_test, y_test, class_names)


def plot_metrics(metrics_list, model, x_test, y_test, class_names):
    if 'Confusion Matrix' in metrics_list:
        with st.expander("Confusion Matrix"):
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names)
            st.pyplot(plt.gcf())
    if 'ROC Curve' in metrics_list:
        with st.expander("ROC Curve"):
            RocCurveDisplay.from_estimator(model, x_test, y_test)
            st.pyplot(plt.gcf())
    if 'Precision-Recall Curve' in metrics_list:
        with st.expander("Precision-Recall Curve"):
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
            st.pyplot(plt.gcf())


def main():
    st.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")

    df = load_data()

    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if classifier == "Support Vector Machine (SVM)":
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma")

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"), key='metric')

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model_results(df, model, metrics)

    # Repeat above but with Logistic Regression
    if classifier == "Logistic Regression":
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key="max_iter")

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"), key='metric_LR')

        if st.sidebar.button("Classify", key="classify_LR"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model_results(df, model, metrics)

    # Repeat above but with Random Forest
    if classifier == "Random Forest":
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", (True, False), key='boostrap')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"), key='metric_RF')

        if st.sidebar.button("Classify", key="classify_RF"):
            st.subheader("Random Forest")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model_results(df, model, metrics)

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)


if __name__ == '__main__':
    st.markdown("""<style>
                body {text-align: center}
                p {text-align: center} 
                button {float: center} 
                </style>""", unsafe_allow_html=True)
    main()
