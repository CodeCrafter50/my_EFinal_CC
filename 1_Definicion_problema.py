#------------------------------------------------------------------#
# UPC - PROGRAMA DE CIENCIA DE DATOS FOR BUSINESS
# CURSO: TALLERES
# TRABAJO FINAL 
# Definición del problema: 
# Se tiene un dataframe con los datos de 1000 pacientes relacionados a la condición de salud (parámetros clínicos) y a los hábitos de consumo de alochol y de ejecrcicio físico, además del estado de su salud cardiaca.
# Se busca crear un modelo que nos permita pronosticar si una persona sufre o no de enfermedad cardiaca (Heart desease).
# Se eligió este caso, por ser uno de los más conociedos en la elaboración de modelos predictivos.
# ALUMNO: CARLOS CALERO
# FECHA: 13.04.2025
#------------------------------------------------------------------#
import subprocess
import sys

# Lista de paquetes necesarios
required_packages = [
    "sweetviz",
    "pandas",
    "numpy",
    "scikit-learn"
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
import sweetviz as sv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder  
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier


# Carga de datos
df = pd.read_csv(r"D:\Python_ws\Talleres\TF\TFinal\data\heart_disease_dataset.csv")

# Describir el dataframe
print(df.describe())


# --------------------------------------------------

# 1) Realizar EDA usando sweetviz
report = sv.analyze(df)

# Mostrar el reporte
#report.show_html("heart_report_encoded.html")
report.show_html("heart_report.html")


# --------------------------------------------------

# 2) Definir por lo menos dos cortes en el dataset para realizar la interpretación de los resultados
# 
# Selección de cariables de corte:
# Debido a que el EDA muestra una correlación significativa entre la variable dependiente "Heart Disease" 
# y las variables independientes "Age" y "Cholesterol", las seleccionaremos para establecer cortes en ambas 
# y realizar un analisis más detallado.

# a) Con la variable "Age" definimos los siguientes Grupos Etarios: ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
df['Age_Group'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, 70, 80], labels=['20-29', '30-39', '40-49', '50-59', '60-69', '70-79'])

# b) Con la variable "Cholesterol" definimos los siguientes Grupos de Riesgo: ['100-149', '150-199', '200-249', '250-299', '300-399', '400-499', '500-599']s 
df['Cholesterol_Group'] = pd.cut(df['Cholesterol'], bins=[100, 150, 200, 250, 300, 400, 500, 600], labels=['100-149', '150-199', '200-249', '250-299', '300-399', '400-499', '500-599'])

# Analizar la distribución de frecuencia de Enfermedad cardiaca ("HeartDisease") en los Grupos Etarios
print(pd.crosstab(df['Age_Group'], df['Heart Disease']))
plt.figure(figsize=(10, 6))
sns.countplot(x='Age_Group', hue='Heart Disease', data=df)
plt.title('Heart Disease vs Age Group')
plt.show()
# Expicación de los resultados obtenidos:
# La gráfica muestra la relación entre los grupos etarios y la presencia de enfermedades cardíacas.
# Los grupos de edad más jóvenes (20-29, 30-39 y 40-49) no tienen incidencia de enfermedades cardíacas, 
# mientras que los grupos mayores (50-59, 60-69 y 70-79) muestran una prevalencia bastante alta, 
# superando el 70% de personas con enfermedad cardiaca dentro de cada grupo etario. 
# Esto sugiere que la edad es un factor muy importante en la salud cardíaca.


# Analizar la distribución de frecuencia de Enfermedad cardiaca ("HeartDisease") en los Grupos de Riesgo
print(pd.crosstab(df['Cholesterol_Group'], df['Heart Disease']))
plt.figure(figsize=(10, 6))
sns.countplot(x='Cholesterol_Group', hue='Heart Disease', data=df)
plt.title('Heart Disease vs Cholesterol Group')
plt.show()
# Expicación de los resultados obtenidos:
# La gráfica muestra la relación entre los grupos de riesgo y la presencia de enfermedades cardíacas.
# Los grupos cuyo nivel de colesterol se encuentra en los rangos: ['100-149', '150-199'] no tienen incidencia de enfermedades cardíacas, 
# mientras que los grupos cuyo nivel de colesterol se encuentra en los rangos: ['200-249', '250-299', '300-399'] muestran una prevalencia importante, 
# superando el 50% de personas con enfermedad cardiaca en cada grupo de riesgo. 
# Esto sugiere que el nivel de colesterol es un factor importante en la salud cardíaca.

