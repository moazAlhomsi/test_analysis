import streamlit as st
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 8)
st.set_option('deprecation.showPyplotGlobalUse', False)

breast_cancer = datasets.load_breast_cancer()
x = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
y = pd.Categorical.from_codes(breast_cancer.target, breast_cancer.target_names)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# -----------------------------------------------------
# ---Logistic Regression---
# -----------------------------------------------------

def logistic_regression():
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    y_pred_test = lr.predict(x_test)
    y_pred_train = lr.predict(x_train)
    

    st.write("____________________________________________________________________________________________________________________")
    st.markdown("""<h1 style="font-family: 'Times New Roman', Times, serif; font-size: 40px; color: #ff4d4d; font-weight: bold;">Model Evaluation</h1>""", unsafe_allow_html=True)
    st.write("____________________________________________________________________________________________________________________")
    st.container()
    col1 , col2 = st.columns((3,2))
    with col1:
        # st.markdown("## Acuracy on test set:")
        st.metric(label="Acuracy on test set", value="{:.2%}".format(round(accuracy_score(y_test, y_pred_test),3)))
        if accuracy_score(y_test, y_pred_test) >= 0.9:
            st.write("The accuracy score is high and shows a great prediction power")

    with col2:
        # st.markdown("## Acuracy on training set")
        # st.write("{:.2%}".format(round(accuracy_score(y_train, y_pred_train),3)))
        st.metric(label="Acuracy on training set", value="{:.2%}".format(round(accuracy_score(y_train, y_pred_train),3)))
        if accuracy_score(y_train, y_pred_train) >= 0.9:
            st.write("The accuracy score is high and shows a great prediction power")

    if accuracy_score(y_train, y_pred_train) - accuracy_score(y_test, y_pred_test) > 0.4:
        st.warning("The model is overfitting")

    elif accuracy_score(y_train, y_pred_train) - accuracy_score(y_test, y_pred_test) < -0.4:
        st.warning("The model is underfitting")

    else:
        st.success("The model is neither overfitting nor underfitting",icon="✅")

    st.write("____________________________________________________________________________________________________________________")
    with st.expander('performance'):
        st.markdown("""<h1 style="font-family: 'Times New Roman', Times, serif; font-size: 40px; color: #ff4d4d; font-weight: bold;">Model Performance</h1>""", unsafe_allow_html=True)
        st.container()
        col1,col2,col3, col4 = st.columns((1,2,2,3))
        with col2:
            st.title("Confusion Matrix")
            st.table(confusion_matrix(y_test, y_pred_test))
        with col1:
            st.write("")
        with col4:
            st.title("Classification Report") 
            st.text(classification_report(y_test, y_pred_test))
        with col3:
            st.write("")
    st.write("____________________________________________________________________________________________________________________")


    


    st.write("____________________________________________________________________________________________________________________")
    with st.expander('plots'):
        st.markdown("""<h1 style="font-family: 'Times New Roman', Times, serif; font-size: 40px; color: #ff4d4d; font-weight: bold;">Plots</h1>""", unsafe_allow_html=True)
        st.container()
        col1, col2 = st.columns(2)
        with col1:
            st.title("ROC Curve")
            fig = plt.plot()
            RocCurveDisplay.from_estimator(lr, x_test, y_test)
            st.pyplot(fig)
        with col2:
            st.write("")
    st.write("____________________________________________________________________________________________________________________")

          

    
    
# -----------------------------------------------------
# ---Random Forest---
# -----------------------------------------------------

def random_forest():
    rf = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42)
    rf.fit(x_train, y_train)
    y_pred_test = rf.predict(x_test)
    y_pred_train = rf.predict(x_train)

    st.write("____________________________________________________________________________________________________________________")
    st.markdown("""<h1 style="font-family: 'Times New Roman', Times, serif; font-size: 40px; color: #ff4d4d; font-weight: bold;">Model Evaluation</h1>""", unsafe_allow_html=True)
    st.write("____________________________________________________________________________________________________________________")
    st.container()
    col1 , col2 = st.columns((3,2))
    with col1:
        # st.markdown("## Acuracy on test set:")
        st.metric(label="Acuracy on test set", value="{:.2%}".format(round(accuracy_score(y_test, y_pred_test),3)))
        if accuracy_score(y_test, y_pred_test) >= 0.9:
            st.write("The accuracy score is high and shows a great prediction power")

    with col2:
        # st.markdown("## Acuracy on training set")
        # st.write("{:.2%}".format(round(accuracy_score(y_train, y_pred_train),3)))
        st.metric(label="Acuracy on training set", value="{:.2%}".format(round(accuracy_score(y_train, y_pred_train),3)))
        if accuracy_score(y_train, y_pred_train) >= 0.9:
            st.write("The accuracy score is high and shows a great prediction power")

    if accuracy_score(y_train, y_pred_train) - accuracy_score(y_test, y_pred_test) > 0.4:
        st.warning("The model is overfitting")

    elif accuracy_score(y_train, y_pred_train) - accuracy_score(y_test, y_pred_test) < -0.4:
        st.warning("The model is underfitting")

    else:
        st.success("The model is neither overfitting nor underfitting",icon="✅")

    st.write("____________________________________________________________________________________________________________________")
    with st.expander('performance'):
        st.markdown("""<h1 style="font-family: 'Times New Roman', Times, serif; font-size: 40px; color: #ff4d4d; font-weight: bold;">Model Performance</h1>""", unsafe_allow_html=True)
        st.container()
        col1,col2,col3, col4 = st.columns((1,2,2,3))
        with col2:
            st.title("Confusion Matrix")
            st.table(confusion_matrix(y_test, y_pred_test))
        with col1:
            st.write("")
        with col4:
            st.title("Classification Report") 
            st.text(classification_report(y_test, y_pred_test))
        with col3:
            st.write("")
    st.write("____________________________________________________________________________________________________________________")

    
    
    
    st.write("____________________________________________________________________________________________________________________")
    with st.expander('plots'):
        st.markdown("""<h1 style="font-family: 'Times New Roman', Times, serif; font-size: 40px; color: #ff4d4d; font-weight: bold;">Plots</h1>""", unsafe_allow_html=True)
        st.container()
        col1, col2 = st.columns(2)
        with col1:
            st.title("ROC Curve")
            fig = plt.plot()
            RocCurveDisplay.from_estimator(rf, x_test, y_test)
            st.pyplot(fig)
        with col2:
            imp = pd.Series(rf.feature_importances_, index=x.columns)  
            st.title("Feature Importance")
            fig = plt.figure()
            imp.nlargest(8).plot(kind='barh').invert_yaxis()
            st.pyplot(fig)
    st.write("____________________________________________________________________________________________________________________")

    

# -----------------------------------------------------    
# ---Naive Bayes---
# -----------------------------------------------------

def naive_bayes():
    nb = GaussianNB()
    nb.fit(x_train, y_train)
    y_pred_test = nb.predict(x_test)
    y_pred_train = nb.predict(x_train)

    st.write("____________________________________________________________________________________________________________________")
    st.markdown("""<h1 style="font-family: 'Times New Roman', Times, serif; font-size: 40px; color: #ff4d4d; font-weight: bold;">Model Evaluation</h1>""", unsafe_allow_html=True)
    st.write("____________________________________________________________________________________________________________________")
    st.container()
    col1 , col2 = st.columns((3,2))
    with col1:
        # st.markdown("## Acuracy on test set:")
        st.metric(label="Acuracy on test set", value="{:.2%}".format(round(accuracy_score(y_test, y_pred_test),3)))
        if accuracy_score(y_test, y_pred_test) >= 0.9:
            st.write("The accuracy score is high and shows a great prediction power")

    with col2:
        # st.markdown("## Acuracy on training set")
        # st.write("{:.2%}".format(round(accuracy_score(y_train, y_pred_train),3)))
        st.metric(label="Acuracy on training set", value="{:.2%}".format(round(accuracy_score(y_train, y_pred_train),3)))
        if accuracy_score(y_train, y_pred_train) >= 0.9:
            st.write("The accuracy score is high and shows a great prediction power")

    if accuracy_score(y_train, y_pred_train) - accuracy_score(y_test, y_pred_test) > 0.4:
        st.warning("The model is overfitting")

    elif accuracy_score(y_train, y_pred_train) - accuracy_score(y_test, y_pred_test) < -0.4:
        st.warning("The model is underfitting")

    else:
        st.success("The model is neither overfitting nor underfitting",icon="✅")

    st.write("____________________________________________________________________________________________________________________")
    with st.expander('performance'):
        st.markdown("""<h1 style="font-family: 'Times New Roman', Times, serif; font-size: 40px; color: #ff4d4d; font-weight: bold;">Model Performance</h1>""", unsafe_allow_html=True)
        st.container()
        col1,col2,col3, col4 = st.columns((1,2,2,3))
        with col2:
            st.title("Confusion Matrix")
            st.table(confusion_matrix(y_test, y_pred_test))
        with col1:
            st.write("")
        with col4:
            st.title("Classification Report") 
            st.text(classification_report(y_test, y_pred_test))
        with col3:
            st.write("")
    st.write("____________________________________________________________________________________________________________________")
    st.write("____________________________________________________________________________________________________________________")
    with st.expander('plots'):
        st.markdown("""<h1 style="font-family: 'Times New Roman', Times, serif; font-size: 40px; color: #ff4d4d; font-weight: bold;">Plots</h1>""", unsafe_allow_html=True)
        st.container()
        col1, col2 = st.columns(2)
        with col1:
            st.title("ROC Curve")
            fig = plt.plot()
            RocCurveDisplay.from_estimator(nb, x_test, y_test)
            st.pyplot(fig)
        with col2:
            st.write("")
    st.write("____________________________________________________________________________________________________________________")

    
    
