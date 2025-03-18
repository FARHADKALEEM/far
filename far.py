import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Title for the app
st.title("Fraud Detection Model")

# File uploaders to upload training and testing data
train_file = st.file_uploader("C:/Users/Hussian computer/Downloads/fraudTrain.csv")
test_file = st.file_uploader("C:/Users/Hussian computer/Downloads/fraudTest.csv")

# Check if both files are uploaded
if train_file is not None and test_file is not None:
    # Load the datasets directly from the uploaded files
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    
    # Combine both datasets
    df = pd.concat([train, test], ignore_index=True)
    
    # Display the combined dataframe
    st.write("Combined Data:")
    st.dataframe(df)
    
    # Check for missing values
    if df.isnull().sum().any():
        st.warning("There are missing values in the dataset.")
        df.fillna(0, inplace=True)  # Optionally fill missing values
    else:
        st.success("No missing values found.")
    
    # Drop 'Unnamed: 0' if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Initialize LabelEncoder
    le = LabelEncoder()
    
    # Apply LabelEncoder only to categorical columns
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = le.fit_transform(df[column])

    # Prepare features (x) and target (y)
    x = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train the model
    model = svm.SVC()
    model.fit(x_train, y_train)

    # Make predictions
    y_predict = model.predict(x_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_predict)

    # Display accuracy
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
    