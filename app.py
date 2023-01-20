import streamlit as st
from streamlit_option_menu import option_menu
from main import model
from src import *
from matplotlib import pyplot as plt

result = 0

def getResult(sentiment, service, food_quality, price, environment, aggregate_operation, defuzzification_method):
        score = model.predict(sentiment, service, food_quality, price, environment, aggregate_operation, defuzzification_method)
        return score


with st.sidebar:
    choose = option_menu("Restaurant Rating System", ["Analyze", "Membership Functions"],
                         icons=['kanban', 'graph-up'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#aaa"},
        "menu-title" : {"font-size": "17px"}
    }
)


if choose == "Analyze":
    st.title("Analyze")

    aggregate_operation = st.selectbox(
        "Aggregate operation",
        ("Max", "Mean"),
    )

    defuzzification_method = st.selectbox(
        "Dufuzzification Method",
        ("centroid", "bisect", "mom", "som", "lom"),
    )

    sentiment = st.slider('Sentiment: Please choose between 0 - 100', 0, 100, value=0)
    st.write("Sentiment:", sentiment)

    service = st.slider('Service: Please choose between 0 - 100', 0, 100, value=0)
    st.write("Service:", service) 

    food_quality = st.slider('Food Quality: Please choose between 0 - 100', 0, 100, value=0)
    st.write("Food Quality:", food_quality) 
    
    price = st.slider('Price: Please choose between 5 - 60 (RM)', 5, 60, value=5)
    st.write("Price:", price) 

    environment = st.slider('Environment: Please choose between 0 - 100', 0, 100, value=0)
    st.write("Environment:", environment) 
    
    if st.button("Analyze"):
        result = getResult(sentiment, service, food_quality, price, environment, aggregate_operation, defuzzification_method)

    st.subheader("Restaurant Rating (0-100): " + str("{:.2f}".format(result)))
    if result != 0:
        model.visualize()

    
if choose == "Membership Functions":
    for variable, mfs in MF.items():
        fig, ax = plt.subplots()
        
        for item, mf in mfs.items():
            ax.plot(mf)

        ax.set_title(variable.name)
        ax.legend(mfs.keys(), loc='upper right')

        st.pyplot(fig)
        
    
