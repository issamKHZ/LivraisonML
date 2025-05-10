import datetime
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Chargement du modèle
with open("modele.pkl", "rb") as f:
    model = pickle.load(f)

# === Définir les encodeurs comme à l’entraînement ===
le_weather = LabelEncoder()
le_weather.fit(['Sunny', 'Sandstorms', 'Cloudy', 'Fog', 'Windy', 'Stormy'])

le_traffic = LabelEncoder()
le_traffic.fit(['High', 'Low', 'Medium', 'Jam'])

le_vehicle = LabelEncoder()
le_vehicle.fit(['motorcycle', 'scooter', 'electric_scooter', 'bicycle'])

le_city = LabelEncoder()
le_city.fit(['Urban', 'Semi-Urban', 'Metropolitian'])

le_time_period = LabelEncoder()
le_time_period.fit(['Morning', 'Afternoon', 'Evening', 'Night'])

ohe_multi = OneHotEncoder(sparse_output=False, drop='first')
ohe_multi.fit(pd.DataFrame({'multiple_deliveries': ['0', '1']}))

# === Fonction de préparation ===
def prepare_input_streamlit(user_input):
    weather_enc = le_weather.transform([user_input['Weatherconditions']])[0]
    traffic_enc = le_traffic.transform([user_input['Road_traffic_density']])[0]
    vehicle_enc = le_vehicle.transform([user_input['Type_of_vehicle']])[0]
    city_enc = le_city.transform([user_input['City']])[0]
    time_period_enc = le_time_period.transform([user_input['Time_Period']])[0]
    multi_ohe = ohe_multi.transform([[user_input['multiple_deliveries']]])[0]
    
    input_vector = np.array(
        [
            user_input['Delivery_person_Age'],
            user_input['Delivery_person_Ratings'],
            weather_enc,
            traffic_enc,
            vehicle_enc,
            city_enc,
            user_input['Hour_order_picked'],
            time_period_enc,
            user_input['distance_km'],
        ] + multi_ohe.tolist()
    ).reshape(1, -1)

    return input_vector

# === Streamlit app ===
st.set_page_config(page_title="Prédiction Livraison", layout="wide")
st.markdown("<h1 style='text-align: center; color: #0E76A8;'>📦 Prédiction du Temps de Livraison</h1>", unsafe_allow_html=True)
st.markdown("## ")

# === Définir l’historique ===
history_file = "pred_history.csv"
if os.path.exists(history_file):
    history_df = pd.read_csv(history_file)
else:
    history_df = pd.DataFrame(columns=[
        "Delivery_person_Age", "Delivery_person_Ratings", "Weatherconditions",
        "Road_traffic_density", "Type_of_vehicle", "multiple_deliveries",
        "City", "Hour_order_picked", "Time_Period", "distance_km",
        "Predicted_time_min"
    ])

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Saisir les informations de livraison")

    rating = st.number_input("Rating du livreur", min_value=1.0, max_value=4.9, step=0.1)
    age = st.number_input("Âge du livreur", min_value=18, max_value=45, step=1)
    distance = st.number_input("Distance (en km)", min_value=0.0, max_value=100.0, step=0.1)

    vehicule = st.selectbox("Type de véhicule", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
    traffic = st.selectbox("La densité de traffique", ["High", "Low", "Medium", "Jam"])
    multi_delivery = st.selectbox("Livraisons multiples", ["Oui", "Non"])
    weather = st.selectbox("Conditions météo", ["Sunny", "Sandstorms", "Cloudy", "Fog", "Windy", "Stormy"])
    city = st.selectbox("Ville", ["Urban", "Semi-Urban", "Metropolitian"])

    heure = st.time_input("Heure de prise de commande", value=datetime.time(12, 0))
    heure_int = heure.hour

    if 5 <= heure_int < 12:
        time_period = 'Morning'
    elif 12 <= heure_int < 17:
        time_period = 'Afternoon'
    elif 17 <= heure_int < 21:
        time_period = 'Evening'
    else:
        time_period = 'Night'

    if st.button("Prédire"):
        user_input = {
            'Delivery_person_Age': age,
            'Delivery_person_Ratings': rating,
            'Weatherconditions': weather,
            'Road_traffic_density': traffic,
            'Type_of_vehicle': vehicule,
            'City': city,
            'Hour_order_picked': heure_int,
            'Time_Period': time_period,
            'distance_km': distance,
            'multiple_deliveries': multi_delivery
        }

        input_vector = prepare_input_streamlit(user_input)
        prediction = model.predict(input_vector)[0]

        st.success(f"⏱️ Temps de livraison estimé : {round(prediction)} minutes")

        # Ajouter la nouvelle prédiction à l’historique
        new_row = pd.DataFrame([{
            "Delivery_person_Age": age,
            "Delivery_person_Ratings": rating,
            "Weatherconditions": weather,
            "Road_traffic_density": traffic,
            "Type_of_vehicle": vehicule,
            "multiple_deliveries": multi_delivery,
            "City": city,
            "Hour_order_picked": heure_int,
            "Time_Period": time_period,
            "distance_km": distance,
            "Predicted_time_min": round(prediction)
        }])

        history_df = pd.concat([new_row, history_df], ignore_index=True).head(10)
        history_df.to_csv(history_file, index=False)

with col2:
    st.subheader("📈 Dernières Prédictions")
    if not history_df.empty:
        st.dataframe(history_df.round(2))
    else:
        st.info("Aucune prédiction encore disponible.")
