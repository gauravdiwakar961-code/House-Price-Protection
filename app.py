import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import time
import pandas as pd
from sklearn.datasets import fetch_california_housing
st.title('House price predicttion using ML🏠')

st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTGcvEdW1IKHjeztlFuZ1pdgax-iGdZdQ5Xtw&s')

data = fetch_california_housing

X = pd.DataFrame(data['data'],
        columns = data['feature_names'])

final_X = X.iloc[:,:-2]
scaler = StandardScaler()
scaled_X = scaler.fit_transform(final_X)
