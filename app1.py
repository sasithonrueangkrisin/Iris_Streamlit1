import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="centered")

st.title("🌸 Iris Flower Classifier")
st.write("แอปพลิเคชันทำนายสายพันธุ์ดอกไอริสด้วย Machine Learning (KNN)")

# โหลดข้อมูลและเทรนโมเดล
iris = load_iris()
X, y = iris.data, iris.target
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)

# Sidebar สำหรับรับค่า Input
st.sidebar.header("📊 Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

# ทำนายผล
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

species_names = iris.target_names
predicted_species = species_names[prediction[0]]
confidence = prediction_proba[0][prediction[0]] * 100

# แสดงผลลัพธ์บนหน้าเว็บ
st.subheader("🔮 Prediction Result")
st.success(f"สายพันธุ์ที่ทำนายได้: **{predicted_species.capitalize()}**")
st.metric(label="Prediction Confidence", value=f"{confidence:.1f}%")

# แสดงกราฟเปรียบเทียบข้อมูลที่กรอก
st.subheader("📈 Features Comparison")
chart_data = pd.DataFrame({
    'Features': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
    'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
})
st.bar_chart(chart_data.set_index('Features'))