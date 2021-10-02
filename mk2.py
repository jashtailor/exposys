# IMPORTING THE NECESSARY LIBRARIES
import pickle

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.models import model_from_json

import streamlit as st
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

st.write("""
# Generic Medical Name
""")

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LOADING THE MODELS
LogReg = pickle.load(open('Logistic Regression (Hyperparameter tuned).pkl', 'rb'))
RFC1 = pickle.load(open('Random Forest Classifier (Hyperparameter tuned).pkl', 'rb'))
SGDC1 = pickle.load(open('Stochastic Gradient Descent (Hyperparameter tuned).pkl', 'rb'))
svm1 = pickle.load(open('Support Vector Machine.pkl', 'rb'))
L_SVC = pickle.load(open('Linear Support Vector Machine.pkl', 'rb')) # this is the Hyperparameter tuned version 
GNB = pickle.load(open('Naive Bayes.pkl', 'rb'))
BNB = pickle.load(open('Bernoulli Naive Bayes.pkl', 'rb'))
MNB1 = pickle.load(open('Multinomial Naive Bayes (Hyperparameter tuned).pkl', 'rb'))
AdaB1 = pickle.load(open('AdaBoost Classifier (Hyperparameter tuned).pkl', 'rb'))
LGBM = pickle.load(open('Light Gradient Boosting Machine.pkl', 'rb'))
GBC1 = pickle.load(open('Gradient Boosting Classifier (Hyperparameter tuned).pkl', 'rb'))

model_lst = [LogReg, RFC1, SGDC1, svm1, L_SVC, GNB, BNB, MNB1, AdaB1, LGBM, GBC1]

json_file = open('network.json', 'r')
loaded_network_json = json_file.read()
json_file.close()
loaded_network = model_from_json(loaded_network_json)
loaded_network.load_weights("network.h5")
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TAKING INPUT FROM USER

with st.form(key='my_form'):
	text_input = st.text_input(label='Enter some text')
	Pregnancies = st.text_input(label="Number of Pregnancies")
	Glucose = st.text_input(label="Glucose levels")
	BloodPressure = st.text_input(label="Blood Pressure levels")
	SkinThickness = st.text_input(label="Skin Thickness")
	Insulin = st.text_input(label="Insulin levels")
	BMI = st.text_input(label="BMI of the patient")
	DiabetesPedigreeFunction = st.text_input(label="Diabetes Pedigree Function")
	Age = st.text_input(label="Age of the patient")
	submit_button = st.form_submit_button(label='Submit')


df1 = pd.DataFrame(
    data=[[Pregnancies,
    Glucose,
    BloodPressure,
    SkinThickness,
    Insulin,
    BMI,
    DiabetesPedigreeFunction,
    Age]],
    columns=['Pregnancies', 
    'Glucose', 
    'BloodPressure', 
    'SkinThickness', 
    'Insulin', 
    'BMI', 
    'DiabetesPedigreeFunction', 
    'Age'])
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MAKING PREDICTION
lst1 = []
for i in model_lst:
    lst1.append(i.predict(df1))
m = network.predict(df1)
lst1.append(np.where(m[0] == max(m[0])))
if lst1.count(1)>lst1.count(0):
    st.write('Patient is likely to be diagnosed with Diabetes')
else:
    st.write('Patient is unlikely to be diagnosed with Diabetes')
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
