import streamlit as st
from app1 import logistic_regression, random_forest, naive_bayes
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json


st.set_page_config(page_title="Breast Cancer", page_icon="⚕️", layout="wide")


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



def lottie(file_path=str):
    with open(file_path, "r") as f:
        return json.load(f)



st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
""", unsafe_allow_html=True)


st.markdown("""<h1 style="font-family: 'Times New Roman', Times, serif; font-size: 40px; color: #ffccff; font-weight: bold;">Breast Cancer Prediction</h1>""", unsafe_allow_html=True)

lotfile = lottie("cancer.json")

    

navi = option_menu(
    menu_title="ML Models",
    options=["Home","Logistic Regression", "Random Forest", "GaussianNB"],
    icons = ["house","dot", "dot", "dot"],
    orientation="horizontal")  

if navi == "Home":
    st.write("""""")
    col1, col2 = st.columns(2)
    with col1:
        st_lottie(
                lotfile,
                speed=1,
                loop=True,
                reverse=False,
                quality="low",
                height=500,
                width=900,
                key=None
            )
    with col2:
        st.write("""
___________________________________________________________________________________________________________________________________



## Breast Cancer Overview:

Breast cancer is a type of cancer that affects the breast tissue, and it is the most common cancer among women worldwide. Early detection and treatment are crucial for survival. Some common symptoms of breast cancer include a lump or thickening in the breast or underarm, change in size or shape of the breast, skin dimpling, nipple discharge, and redness or scaling of the nipple or skin (National Cancer Institute, 2021).

The exact cause of breast cancer is unknown, but several risk factors have been identified, including age, family history of breast cancer, exposure to estrogen, genetic mutations, and personal history of breast or ovarian cancer (National Cancer Institute, 2021).

Treatment options for breast cancer depend on the stage and type of cancer, and may include surgery, radiation therapy, chemotherapy, hormonal therapy, and targeted therapy (National Cancer Institute, 2021).

Reference:

National Cancer Institute. (2021). Breast Cancer Treatment (PDQ) - Health Professional Version. Retrieved from https://www.cancer.gov/types/breast/patient/breast-treatment-pdq""")


 
if navi == "Logistic Regression":
    logistic_regression()
    

if navi == "Random Forest":
    random_forest()


if navi == "GaussianNB":
    naive_bayes()
    

