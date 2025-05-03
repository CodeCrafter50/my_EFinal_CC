#------------------------------------------------------------------#
# UPC - PROGRAMA DE CIENCIA DE DATOS FOR BUSINESS
# CURSO: TALLERES
# TRABAJO PARCIAL (PREGUNTAS 3, 4 Y 5)
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
    "scikit-learn",
    "xgboost",
    "optuna"
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
import pickle
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder  
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Carga de datos
df = pd.read_csv(r"D:\Python_ws\Talleres\TF\TFinal\data\heart_disease_dataset.csv")


# 3) PREPROCESAMIENTO - CONVERSIÓN DE VARIABLES CATEGÓRICAS A NUMÉRICAS USANDO PIPELINES DE SKLEARN

# SELECCIONAMOS MEDIANTE ALGORITMO LA VARIABLE OBJETIVO (Traget)
# El algoritmo consistirá en seleccionar la variable numérica que tiene la correlación más alta con respecto a las otras variables
# Calcular correlaciones
correlations = df.select_dtypes(include=np.number).corr()
# Seleccionar la columna con la correlación más alta con otras variables
target_column = correlations.abs().sum().idxmax()
print(f"Variable objetivo identificada: {target_column}")


# Seleccionamos las variables categóricas y numéricas
#categorical = ['Gender', 'Family History', 'Diabetes', 'Obesity', 'Exercise Induced Angina', 'Smoking', 'Alcohol Intake', 'Chest Pain Type']
#numerical = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical = [col for col in df.select_dtypes(include=['int64', 'float64']).columns.tolist() if col != target_column] # Excluimos la variable target de la lista de variables numéricas


# ESCALADO DE VARIABLES NUMÉRICAS (JUSTIFICACIÓN)
# Debido a que las variables numéricas muestran una distribución uniforme,
# usaremos la normalización (Min-Max scaling) para no cambiar la forma de distribucuión de los datos.
# Al escalar también evitaremos que las variable con rangos amplios dominen.
# Para las variables categóricas, usaremos OneHotEncoder para convertirlas en variables dummy.

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ]), numerical),

        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical)
    ]
)


# 4) MODELAMIENTO CON 5 MODELOS DE CLASIFICACIÓN

# Primero dividimos los datos en características (X) y variable objetivo (y)
X = df.drop(columns=[target_column], axis=1)
y = df[[target_column]]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

print(X.describe())
print(X.head())

print(y.describe())
print(y.head())


# Autmatizamos el modelado de 5 modelos de clasificación diferentes usando pipelines de sklearn:
# 1. Logistic Regression
# 2. Random Forest
# 3. K-Nearest Neighbors
# 4. Support Vector Machine
# 5. Gradient Boosting (XGBoost o similar)

models = {
    "LogReg": LogisticRegression(),
    "RF": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "XGB": XGBClassifier()
}

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    pipeline.fit(X_train, y_train.values.ravel()) # Asegurar que y_train sea un array 1D
    y_pred = pipeline.predict(X_test)
    print(f"\nModelo: {name}")
    print(classification_report(y_test, y_pred))


# 5) OPTIMIZAR HIPERPARÁMETROS DE LOS MODELOS UNSANDO OPTUNA

# Debido a que el accuracy de los modelos "XGBoost" (XGB) y "RandomForest" (RF) están muy cercanos a 1, lo cuál
# estaría indicando un sobreajuste, se buscará optimizar
# solamente los hiperparámetros de los modelos de "Logistic Regression" (LogReg), 
# "K-Nearest Neighbors" (KNN) y "Support Vector Machine" (SVM).

import optuna


def objective(trial):
    # Definir el espacio de búsqueda de hiperparámetros para cada modelo
    # model_name = trial.suggest_categorical("model", ["LogReg", "RF", "KNN", "SVM", "XGB"])
    model_name = trial.suggest_categorical("model", ["LogReg", "KNN", "SVM"])

    if model_name == "LogReg":
        C = trial.suggest_float("C", 1e-10, 1e10, log=True)
        model = LogisticRegression(C=C, max_iter=10000)
#    elif model_name == "RF":
#        n_estimators = trial.suggest_int("n_estimators", 50, 200)
#        max_depth = trial.suggest_int("max_depth", 10, 100)
#        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
#        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    elif model_name == "KNN":
        n_neighbors = trial.suggest_int("n_neighbors", 3, 15)
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_name == "SVM":
        C = trial.suggest_float("C", 1e-10, 1e10, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        model = SVC(C=C, gamma=gamma, kernel=kernel)
#    elif model_name == "XGB":
#        n_estimators = trial.suggest_int("n_estimators", 50, 200)
#        max_depth = trial.suggest_int("max_depth", 3, 10)
#        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
#        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42, use_label_encoder=False, eval_metric='logloss')

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    pipeline.fit(X_train, y_train.values.ravel()) # Asegurar que y_train sea un array 1D
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=15, show_progress_bar=True) # Ajusta el número de trials según sea necesario (se usarán 20 trials)

print("Mejores hiperparámetros:", study.best_params)
print("Mejor precisión:", study.best_value)

# Crear el directorio (si no existe) para almacenar el mejor modelo
model_dir = r"D:\Python_ws\Talleres\TF\TFinal\models"
os.makedirs(model_dir, exist_ok=True)

# Entrenar el modelo SVM con los mejores hiperparámetros encontrados
best_params = study.best_params
if best_params["model"] == "SVM":
    best_model = SVC(C=best_params["C"], gamma=best_params["gamma"], kernel=best_params["kernel"])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', best_model)])
    pipeline.fit(X_train, y_train.values.ravel())

    # Guardar el modelo entrenado en un archivo .pkl usando pickle
    model_path = os.path.join(model_dir, "best_svm_model.pkl")
    with open(model_path, 'wb') as file:
        pickle.dump(pipeline, file)
    print(f"Modelo SVM guardado en: {model_path}")

elif best_params["model"] == "LogReg":
    best_model = LogisticRegression(C=best_params["C"], max_iter=10000)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', best_model)])
    pipeline.fit(X_train, y_train.values.ravel())

    # Guardar el modelo entrenado en un archivo .pkl usando pickle
    model_path = os.path.join(model_dir, "best_logreg_model.pkl")
    with open(model_path, 'wb') as file:
        pickle.dump(pipeline, file)
    print(f"Modelo Logistic Regression guardado en: {model_path}")

elif best_params["model"] == "KNN":
    best_model = KNeighborsClassifier(n_neighbors=best_params["n_neighbors"])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', best_model)])
    pipeline.fit(X_train, y_train.values.ravel())

    # Guardar el modelo entrenado en un archivo .pkl usando pickle
    model_path = os.path.join(model_dir, "best_knn_model.pkl")
    with open(model_path, 'wb') as file:
        pickle.dump(pipeline, file)
    print(f"Modelo KNN guardado en: {model_path}")


