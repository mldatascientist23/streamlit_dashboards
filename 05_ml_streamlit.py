import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# App heading
st.write('''
         # Explore different ML models and datasets!
         Daikhtay hen which one is best
         ''')

st.write('---')

# Sidebar
st.sidebar.header('Select the options below')

# Sidebar - Dataset selection
dataset_name = st.sidebar.selectbox(
    'Select the dataset', ('Iris', 'Breast Cancer', 'Wine')
)

# Sidebar - Model selection
classifier_name = st.sidebar.selectbox(
    'Select the classifier', ('KNN', 'SVM', 'Random Forest')
)

# Function to load datasets
def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x = data.data
    y = data.target
    return x, y

# Load the dataset
X, y = get_dataset(dataset_name)

# Display dataset shape and number of classes
st.write('Shape of dataset: ', X.shape)
st.write('Number of classes: ', len(np.unique(y)))

# Function to add parameter UI for classifiers
def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

# Add classifier parameters
params = add_parameter_ui(classifier_name)

# Function to get the classifier
def get_classifier(classifier_name, params):
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'],
                                     max_depth=params['max_depth'], random_state=1234)
    return clf

# Get the classifier
clf = get_classifier(classifier_name, params)

# Check if classifier is properly created
if clf is None:
    st.write("Error: The classifier could not be created.")
else:
    # Split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Predict the test set
    y_pred = clf.predict(X_test)

    # Check the accuracy
    acc = accuracy_score(y_test, y_pred)
    st.write(f'Classifier: {classifier_name}')
    st.write(f'Accuracy: {acc}')

    # Plot the dataset
    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2,
                c=y, alpha=0.8,
                cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()

    st.pyplot(fig)