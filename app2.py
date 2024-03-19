import os
import streamlit as st
from apikey import apikey
import pandas as pd
import langchain
from langchain.llms import OpenAI 
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv

# OpenAI key
os.environ['OPENAI_API_KEY'] = apikey
load_dotenv(find_dotenv())

# Title
st.title("Daisy🌼")
st.write("👋Hello there! I'm Daisy, your friendly AI assistant, ready to dive deep into the world of data and unlock insights just for you.")

# Sidebar
with st.sidebar:
    st.write('*Your Data Science Adventure here starts with csv file*')
    st.image("ds.png", width=250)

    st.write("🌼 Daisy is a user-friendly data exploration and analysis tool developed with Streamlit! 📊 With Daisy, users can easily upload CSV files to uncover insights from their data. Daisy guides users through exploratory data analysis (EDA) step by step, helping them define their questions, clean and organize datasets, explore data visually 📈 and statistically, identify outliers and anomalies 🕵️‍♂️, perform feature engineering ✨, and analyze correlations between numeric columns. 💡")

    st.divider()
    st.caption("<p style='text-align:center'>made with 🩵 by Advika</p>", unsafe_allow_html=True)

# Function to load and display data
def load_and_display_data():
    user_csv = st.file_uploader("Upload your file here", type="csv")
    if user_csv is not None:
        st.write("File uploaded successfully!")
        try:
            user_csv.seek(0)
            df = pd.read_csv(user_csv, low_memory=False)
            
            # Display data head
            st.subheader('Data Head')
            st.write(df.head())

            # Display data summary
            st.subheader('Data Summary')
            st.write(df.describe())
             # OpenAI language model
            llm = OpenAI(temperature=0)
            pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True)

            # Display EDA steps with buttons
            st.subheader('Exploratory Data Analysis Steps')
            # Initialize the key in session state
            if 'clicked' not in st.session_state:
                st.session_state.clicked = False

            # Function to update the value in session state
            def clicked():
                st.session_state.clicked = True

            start_button = st.button("Let's get started", on_click=clicked)

            if st.session_state.clicked:
          
               
                eda_steps = [
                    "Define the problem or question",
                    "Visualise",
                    "Clean and organize the data",
                    "Explore the data",
                    "Identify outliers and anomalies",
                    "Perform feature engineering",
                    "correlation analysis"
                ]
                
                selected_steps = st.multiselect("Select EDA steps to perform:", eda_steps)
                
                if selected_steps:
                    for step in selected_steps:
                        st.subheader(step)
                        if step == "Define the problem or question":
                           
                            st.write("The first step in EDA is to clearly define the problem or question.")
                            st.write(pandas_agent.run("Define the problem or question in the context of the uploaded data."))
                            
                        elif step == "correlation analysis":
                            st.write("correlation between numeric columns")
                            st.write(pandas_agent.run("correlation between numeric columns"))
                            st.code(pandas_agent.run("code to Calculate the correlation between numeric columns"))
                        elif step == "Visualise":
                            st.write("Visualisations:")
                            st.bar_chart(df)
                        elif step == "Clean and organize the data":
                            st.write("Clean and organize the data before analysis.")
                            
                            st.write(pandas_agent.run("Clean and organize the data in the context of the uploaded data."))
                            st.code(pandas_agent.run("code to clean and organnise the data  in the context of the uploaded data."))
                        elif step == "Explore the data":
                            st.write("Visually and statistically analyze the data.")

                            st.write(pandas_agent.run("Explore the data in the context of the uploaded data."))
                            st.code(pandas_agent.run("code to visualise the data  in the context of the uploaded data."))
                        elif step == "Identify outliers and anomalies":
                            st.write("Identify outliers and anomalies in the data.")
                            st.write(pandas_agent.run("Identify outliers and anomalies in the context of the uploaded data."))
                            st.code(pandas_agent.run("code to identify and remove outliers the data  in the context of the uploaded data."))
                        elif step == "Perform feature engineering":
                            st.write("Create new features or transform existing ones.")
                            st.write(pandas_agent.run("Perform feature engineering in the context of the uploaded data."))
            
                            st.code(pandas_agent.run("code to perform feature engineering on the data  in the context of the uploaded data."))
            # Function to handle user questions
            def func_quest():
                df_info = pandas_agent.run(usd)
                st.write(df_info)
                return
            def function_question_variable(df, pandas_agent, user_question_variable):
                st.line_chart(df, y=[user_question_variable])
               
                normality = pandas_agent.run("Check for normality or specify any transformation needed")
                st.write(normality)
                outliers = pandas_agent.run("Assess the presence of outliers")
                st.write(outliers)
                trends = pandas_agent.run("Analyse trends, seasonality, and patterns")
                st.write(trends)
                missing_values = pandas_agent.run("Determine the extent of missing values")
                st.write(missing_values)
            st.header('Variable of Study')
            # Call function to perform further analysis on a specific variable
            user_ques = st.text_input('Which variable are you interested in?')
            if user_ques:
                function_question_variable(df, pandas_agent, user_ques)

                st.subheader('Further Study')
                usd = st.text_input("Is there anything else you wish to know?")
                if usd is not None and usd not in ("", "no", "No", "NO"):
                    func_quest()
                if usd in ("No", "NO", "no"):
                    st.write("")

                    if usd:
                        st.divider()
                        st.header("Data Science Problem")
                        st.write("Let's move to a business problem")
                        prompt = st.text_input("Add your prompt here")
                        if prompt:
                            res = llm(prompt)

        except Exception as e:
            st.error(f"Error occurred: {e}")

# Run the main function
load_and_display_data()
