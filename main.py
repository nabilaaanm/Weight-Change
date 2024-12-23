import pickle
import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats

# Load pipeline
with open('model-weight-change.pkl', 'rb') as f:
    pipeline = pickle.load(f)

encoder = pipeline['encoder']
scaler = pipeline['scaler']
kmeans = pipeline['kmeans']

st.title('Weight Clustering Application')

col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
    Current_Weight = st.number_input('Berat Saat ini (lbs)', min_value=0.0, step=0.1)
    Daily_Calories_Consumed = st.number_input('Konsumsi Kalori Harian', min_value=0, step=100)

with col2:
    Daily_Caloric_Surplus = st.number_input('Surplus/Defisit Kalori Harian', min_value=0.0, step=0.1)
    Physical_Activity_Level = st.selectbox('Tingkat Aktivitas Fisik', ['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active']) 
    Sleep_Quality = st.selectbox('Kualitas Tidur', ['Poor', 'Fair', 'Good', 'Excellent']) 

Stres_Level = st.number_input('Tingkat Stress', min_value=0, max_value=10, step=1, format="%d")


if st.button('Predict Cluster'):
    gender_code = 'M' if Gender == 'Male' else 'F'
    data_baru = pd.DataFrame([{
        'Gender': gender_code,
        'Current Weight (lbs)': Current_Weight,
        'Daily Calories Consumed': Daily_Calories_Consumed,
        'Daily Caloric Surplus/Deficit': Daily_Caloric_Surplus,
        'Physical Activity Level': Physical_Activity_Level,
        'Sleep Quality': Sleep_Quality,
        'Stress Level': Stres_Level
    }])

    data_baru['Gender'] = data_baru['Gender'].apply(lambda x: encoder['Gender'].transform([x])[0] if x in encoder['Gender'].classes_ else -1)
    data_baru['Physical Activity Level'] = data_baru['Physical Activity Level'].replace({"Sedentary": 0, "Lightly Active": 1, "Moderately Active": 2, "Very Active": 3})
    data_baru['Sleep Quality'] = data_baru['Sleep Quality'].replace({"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3})

    data_baru_scaled = scaler.transform(data_baru)
    cluster_pred = kmeans.predict(data_baru_scaled)

    if cluster_pred[0] == 0:
        cluster_label = 'PENAMBAHAN BERAT BADAN'
    elif cluster_pred[0] == 1:
        cluster_label = 'PENURUNAN BERAT BADAN'

    st.success(f'Data Baru Merupakan Individu dengan Fokus: {cluster_label}')
    distances = np.linalg.norm(kmeans.cluster_centers_ - data_baru_scaled, axis=1)

    cluster_labels = {
        0: 'kelompok yang berfokus pada penambahan berat badan',
        1: 'kelompok yang berfokus pada penurunan berat badan'
    }

    for i, distance in enumerate(distances):
        st.write(f"Jarak data baru ke {cluster_labels[i]}: {distance:.2f}")