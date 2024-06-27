import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make containers
header = st.container()
data_sets = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title('Titanic App')
    st.text('In this project we will work on Titanic data analysis and prediction.')
    
    # Import dataset
    df = sns.load_dataset('titanic')
    df = df.dropna(subset=['age', 'fare', 'embarked', 'sex', 'class', 'who', 'survived'])
    st.write(df.head(10))
    
    st.subheader('Gender distribution')
    st.bar_chart(df['sex'].value_counts())

    st.subheader('Class distribution')
    st.bar_chart(df['class'].value_counts())

    st.subheader('Sample Age distribution')
    st.bar_chart(df['age'].sample(10))

with data_sets:
    st.header('Titanic Dataset Insights')
    st.text('We will work with the Titanic dataset.')

with features:
    st.header("App Features:")
    st.text('These are the features of our app:')
    st.markdown('1. **Feature 1:** This feature shows gender distribution.')
    st.markdown('2. **Feature 2:** This feature shows class distribution.')

with model_training:
    st.header("Model Training and Evaluation")
    st.text('We will train a RandomForest model to predict survival.')

    # Creating columns
    input, display = st.columns(2)

    # Slider for max_depth
    max_depth = input.slider('Select max_depth for the RandomForest:', min_value=10, max_value=100, value=20, step=5)

    # Selectbox for n_estimators
    n_estimators = input.selectbox("Select number of trees for RandomForest:", options=[50, 100, 200, 300, 'No Limit'])
    if n_estimators == 'No Limit':
        n_estimators = None  # Default to None to use the default value of RandomForestRegressor

    # Adding list of features
    input.write("Available features: " + ", ".join(df.columns))

    # Input features from user
    input_features = input.text_input("Which feature should we use for prediction?", 'age')

    if input_features not in df.columns:
        st.error(f"The feature '{input_features}' does not exist in the dataset. Please select a valid feature.")
    else:
        # Machine Learning Model
        model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators if n_estimators else 100)

        # Define X and y
        X = df[[input_features]]
        y = df['fare']

        # Fit our model
        model.fit(X, y)
        pred = model.predict(X)

        # Display metrics
        display.subheader("Mean Absolute Error of the model is:")
        display.write(mean_absolute_error(y, pred))

        display.subheader("Mean Squared Error of the model is:")
        display.write(mean_squared_error(y, pred))

        display.subheader("R Squared score of the model is:")
        display.write(r2_score(y, pred))