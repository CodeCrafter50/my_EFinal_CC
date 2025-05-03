#------------------------------------------------------------------#
# UPC - PROGRAMA DE CIENCIA DE DATOS FOR BUSINESS
# CURSO: TALLERES
# TRABAJO PARCIAL (DESPLIEGUE EN WEB PARA PUNTAJE ADICIONAL)
# ALUMNO: CARLOS CALERO
# FECHA: 13.04.2025
#------------------------------------------------------------------#
import subprocess
import sys

# Lista de paquetes necesarios
required_packages = [
    "streamlit",
    "scikit-learn",
    "pandas"
]

# Instalar paquetes automáticamente si no están presentes
def install_packages(packages):
    for package in packages:
        try:
            __import__(package)  # Intentar importar el paquete
        except ImportError:
            print(f"Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Instalar los paquetes necesarios
install_packages(required_packages)

# Importar paquetes después de la instalación 
import streamlit as st
import pickle
import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Obtener el nombre del archivo .pkl en el sub-directorio "/models" 
def get_pkl_file():
    for file in os.listdir('./models'):
        if file.endswith('.pkl'):
            return file
    return None

pkl_file = get_pkl_file()

# Mostrar el resultado de la búsqueda del archivo .pkl
current_directory = os.getcwd()
# if pkl_file:
#     st.sidebar.write(f"Archivo .pkl detectado: {current_directory}\models\{pkl_file}")
# else:
#     st.sidebar.write(f"Directorio de búsqueda: {current_directory}\models")
#     st.sidebar.error("No se detectó ningún archivo .pkl en el directorio de búsqueda. Finalizando la ejecución.")
#     st.stop()

#Extraer los archivos pickle
pkl_file = os.path.join(current_directory, 'models', pkl_file)
with open(pkl_file, 'rb') as lr:
    best_model = pickle.load(lr)


#print(best_model.named_steps['preprocessor'].feature_names_in_)


# Asignar un nombre al modelo dependiendo del archivo .pkl encontrado
nombre_modelo = None
if "best_svm_model.pkl" in pkl_file:
    nombre_modelo = "Support Vector Machine"
elif "best_logreg_model.pkl" in pkl_file:
    nombre_modelo = "Logistic Regression"
elif "best_knn_model.pkl" in pkl_file:
    nombre_modelo = "K-Nearest Neighbors"
else:
    nombre_modelo = "Modelo desconocido"

#st.sidebar.write(f"Modelo detectado: {nombre_modelo}")


#Funcion mostrar el resultado del pronóstico 
def classify(num):
    if num == 0:
        return 'Sí tiene enfermedad cardíaca'
    elif num == 1:
        return 'No tiene enfermedad cardíaca'
    else:
        return 'N/D. No se puede determinar la existencia de enfermedad cardíaca'


def main():
    #titulo
    st.title('Modelamiento de "Heart Diseases"')
    #titulo de sidebar
    st.sidebar.header('User Input Parameters')

    #funcion para poner los parametros en el sidebar
    def user_input_parameters():

        age = st.sidebar.slider('Age', 25, 79, 52)

        gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
        # Convertir gender a valor numérico
        gender = 1 if gender == 'Male' else 0

        cholesterol = st.sidebar.slider('Cholesterol', 150, 349, 250)

        blood_pressure = st.sidebar.slider('Blood Pressure', 90, 179, 135)

        heart_rate = st.sidebar.slider('Heart Rate', 60, 99, 79)

        smoking = st.sidebar.selectbox('Smoking', ['Current', 'Former', 'Never'])
        # Convertir smoking a valor numérico
        smoking = {'Current': 2, 'Former': 1, 'Never': 0}[smoking]

        alcohol_intake = st.sidebar.selectbox('Alcohol Intake', ['Heavy', 'Moderate', 'None'])
        # Convertir alcohol_intake a valor numérico
        alcohol_intake = {'Heavy': 2, 'Moderate': 1, 'None': 0}[alcohol_intake]

        exercise_hours = st.sidebar.slider('Exercise Hours', 0, 9, 4)

        family_history = st.sidebar.selectbox('Family History', ['Yes', 'No'])
        # Convertir family_history a valor numérico
        family_history = 1 if family_history == 'Yes' else 0

        diabetes = st.sidebar.selectbox('Diabetes', ['Yes', 'No'])
        # Convertir diabetes a valor numérico
        diabetes = 1 if diabetes == 'Yes' else 0

        obesity = st.sidebar.selectbox('Obesity', ['Yes', 'No'])
        # Convertir obesity a valor numérico
        obesity = 1 if obesity == 'Yes' else 0

        stress_level = st.sidebar.slider('Stress Level', 1, 10, 5)

        blood_sugar = st.sidebar.slider('Blood Sugar', 70, 199, 135)

        exercise_induced_angina = st.sidebar.selectbox('Exercise Induced Angina', ['Yes', 'No'])
        # Convertir exercise_induced_angina a valor numérico
        exercise_induced_angina = 1 if exercise_induced_angina == 'Yes' else 0

        chest_pain_type = st.sidebar.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Asymptomatic', 'Non-anginal Pain'])
        # Convertir chest_pain_type a valor numérico
        chest_pain_type = {
            'Typical Angina': 0,
            'Atypical Angina': 1,
            'Asymptomatic': 2,
            'Non-anginal Pain': 3
        }[chest_pain_type]

        data = {'Age': age,
                'Gender': gender,
                'Cholesterol': cholesterol,
                'Blood Pressure': blood_pressure,
                'Heart Rate': heart_rate,
                'Smoking': smoking,
                'Alcohol Intake': alcohol_intake,
                'Exercise Hours': exercise_hours,
                'Family History': family_history,
                'Diabetes': diabetes,
                'Obesity': obesity,    
                'Stress Level': stress_level,   
                'Blood Sugar': blood_sugar, 
                'Exercise Induced Angina': exercise_induced_angina,
                'Chest Pain Type': chest_pain_type
                }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_parameters()

    # Mostrar el modelo con la mejor precisión detectada
    option = [nombre_modelo] 
    model = st.sidebar.selectbox('Modelo con la mejor precisión detectada', option)

    st.markdown("### **Alumno: Carlos Calero**\n### **Curso: Talleres**")
    
    st.write("")
    st.write("")
    st.subheader('Parámetros de entrada del usuario') 
    st.subheader(model)
    st.write(df)

    if st.button('RUN'):
        st.success(classify(best_model.predict(df)))

if __name__ == '__main__':
    main()
