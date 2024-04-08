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
def load_data(df):
    data = pd.read_csv(df)
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


def model_results(df, class_names, model, metrics, container):

    x_train, x_test, y_train, y_test = split(df)
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    y_pred = model.predict(x_test)

    col1, col2, col3 = container.columns(3)
    col1.metric("Accuracy", f'{accuracy:.2f}')
    col2.metric("Precision", f'{precision_score(y_test, y_pred, labels=class_names):.2f}')
    col3.metric("Recall", f'{recall_score(y_test, y_pred, labels=class_names):.2f}')
    st.divider()

    if 'Confusion Matrix' in metrics:
        with container.expander("Confusion Matrix"):
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names)
            st.pyplot(plt.gcf())
    if 'ROC Curve' in metrics:
        with container.expander("ROC Curve"):
            RocCurveDisplay.from_estimator(model, x_test, y_test)
            st.pyplot(plt.gcf())
    if 'Precision-Recall Curve' in metrics:
        with container.expander("Precision-Recall Curve"):
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
            st.pyplot(plt.gcf())


def main():
    st.sidebar.title("Binary Classification")
    st.sidebar.divider()
    st.sidebar.radio("label", ["Default", "Custom"], label_visibility='collapsed', horizontal=True, key="custom")
    st.sidebar.divider()

    container = st.container()

    if st.session_state.custom == "Custom":
        custom_holder = st.empty()
        with custom_holder.container():
            st.title("Upload Custom Binary Dataset")
            st.markdown('#')
            st.markdown('#')
            col1, col2 = st.columns(2)
            class_name1 = col1.text_input('Classification 1')
            class_name2 = col2.text_input('Classification 2')
            class_names = [class_name1, class_name2]
            st.markdown('#')
            st.markdown('#')

            data = st.file_uploader("Choose a file")
        if data is not None and class_name1 != "" and class_name2 != "":
            custom_holder.empty()
            container.title(f"{class_name1} or {class_name2}?")
            container.markdown('#')
            dataset_holder = st.empty()
            with dataset_holder.expander("Data Set"):
                df = load_data(data)
                st.subheader(f"{class_name1} or {class_name2} Data Set")
                st.write(df)
    else:
        container.title("Are the mushrooms edible or poisonous? 🍄")
        container.markdown('#')
        df = load_data(userDir + '/mushrooms.csv')
        class_names = ['edible', 'poisonous']
        dataset_holder = st.empty()
        with dataset_holder.expander("Data Set"):
            st.subheader("Mushroom Data Set (Classification)")
            st.write(df)

    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))
    st.sidebar.divider()

    if classifier == "Support Vector Machine (SVM)":
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        st.sidebar.divider()

        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), horizontal=True, key="kernel")
        st.sidebar.divider()
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), horizontal=True, key="gamma")
        st.sidebar.divider()
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"), key='metric')
        st.sidebar.divider()

        if st.sidebar.button("Classify", key="classify"):
            container.subheader("Support Vector Machine (SVM) Results")
            container.divider()
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model_results(df, class_names, model, metrics, container)

    # Repeat above but with Logistic Regression
    if classifier == "Logistic Regression":
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        st.sidebar.divider()
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key="max_iter")
        st.sidebar.divider()
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"), key='metric_LR')
        st.sidebar.divider()

        if st.sidebar.button("Classify", key="classify_LR"):
            container.subheader("Logistic Regression Results")
            container.divider()
            model = LogisticRegression(C=C, max_iter=max_iter)
            model_results(df, model, metrics, container)

    # Repeat above but with Random Forest
    if classifier == "Random Forest":
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        st.sidebar.divider()
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        st.sidebar.divider()
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", (True, False), horizontal=True, key='boostrap')
        st.sidebar.divider()
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"), key='metric_RF')
        st.sidebar.divider()

        if st.sidebar.button("Classify", key="classify_RF"):
            container.subheader("Random Forest")
            container.divider()
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model_results(df, model, metrics, container)


if __name__ == '__main__':
    st.set_page_config(page_icon='🍄', initial_sidebar_state='expanded')
    st.markdown("""<style>
                body {text-align: center}
                p {text-align: center} 
                button {float: center} 
                [data-testid=stVerticalBlock]{
                    gap: 0rem
                }
                [data-testid=stHorizontalBlock]{
                    gap: 0rem
                }
                [data-testid=stForm] [data-testid=stHorizontalBlock] 
                [data-testid=stHorizontalBlock] [data-testid=column]
                {
                    width: calc(25% - 1rem) !important;
                    flex: 1 1 calc(25% - 1rem) !important;
                    min-width: calc(20% - 1rem) !important;
                }
                [data-testid=stSidebarUserContent] {
                    margin-top: -75px;
                }
                .block-container {
                    padding-top: 1.3rem;
                    padding-bottom: 5rem;
                }
                hr:first-child {
                    margin-top: -0.1px;
                }
                [data-testid="stMetricLabel"] {
                    justify-content: center;
                }
                .stRadio [role=radiogroup]{
                    align-items: center;
                    justify-content: center;
                }
                [data-testid="stMetric"] label {
                    width: fit-content;
                    margin: auto;
                }
                </style>""", unsafe_allow_html=True)
    hide_streamlit_style = """ <style>
              MainMenu {visibility: hidden;}
              header {visibility: hidden;}
              footer {visibility: hidden;}
              </style>"""
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    main()
