import os
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Ilova sarlavhasi
st.title("Bemorga dori tavsiya qilish tizimi")

# Foydalanuvchi ma'lumotlarini kiritish
age = st.number_input("Yoshingizni kiriting:", min_value=1, max_value=120, step=1)

gender = st.selectbox("Jinsingizni tanlang:", ["Male", "Female"])

blood_pressure = st.selectbox("Qon bosimi darajasini tanlang:", ["Low", "High", "Normal"])

Cholesterol = st.selectbox("Xolesterin darajasini tanlang:", ["Normal", "High"])

na_to_k = st.number_input("Natriy-Kaliy nisbatini kiriting:", format="%.2f")

# Tugma qo'shish
if st.button("Bashorat qilish"):
    # Ma'lumotlarni raqamlashtirish
    label_encoder = LabelEncoder()
    gender_encoded = label_encoder.fit_transform([gender])[0]
    blood_pressure_encoded = label_encoder.fit_transform([blood_pressure])[0]
    cholesterol_encoded = label_encoder.fit_transform([Cholesterol])[0]

    # Kiritilgan ma'lumotlarni massivga o'zgartirish
    features = np.array([[age, gender_encoded, blood_pressure_encoded, cholesterol_encoded, na_to_k]])

    # Modelni yuklash
    model_path = os.path.join(os.getcwd(), "decision_tree_model (1).pkl")
    if not os.path.exists(model_path):
        st.error(f"Model fayli topilmadi: {model_path}")
    else:
        with open(model_path, 'rb') as file:
            decision_tree_model = pickle.load(file)

        # Modelga ma'lumotlarni uzatish va bashorat qilish
        prediction = decision_tree_model.predict(features)

        # Natijani chiqarish
        st.success(f"Bashorat: Sizga quyidagi dori mos keladi: {prediction[0]}")
