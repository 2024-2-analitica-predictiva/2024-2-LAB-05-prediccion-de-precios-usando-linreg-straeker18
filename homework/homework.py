#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import pandas as pd 

#importo dataset de prueba
test_data = pd.read_csv(
    "files/input/test_data.csv.zip",
    index_col=False,
    compression="zip",
)
# importo dataset de entrenamiento
train_data = pd.read_csv(
    "files/input/train_data.csv.zip",
    index_col=False,
    compression="zip",
)

# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
test_data['Age']=2021-test_data['Year']
train_data['Age']=2021-train_data['Year']


# - Elimine las columnas 'Year' y 'Car_Name'.
test_data=test_data.drop(columns=['Year','Car_Name'])
train_data=train_data.drop(columns=['Year','Car_Name'])

# %%
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#

x_train=train_data.drop(columns="Present_Price")
y_train=train_data["Present_Price"]


x_test=test_data.drop(columns="Present_Price")
y_test=test_data["Present_Price"]


x_test.head()

# %%
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.feature_selection import f_regression,SelectKBest
import json
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,median_absolute_error
from sklearn.model_selection import GridSearchCV
import pickle
import os
import gzip


#Columnas categoricas
categorical_features=['Fuel_Type','Selling_type','Transmission']
numerical_features= [col for col in x_train.columns if col not in categorical_features]

#preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features),
        ('scaler',MinMaxScaler(),numerical_features),
    ],
)

#pipeline
pipeline=Pipeline(
    [
        ("preprocessor",preprocessor),
        ('feature_selection',SelectKBest(f_regression)),
        ('classifier', LinearRegression())
    ]
)

# %%



param_grid = {
    'feature_selection__k':range(1,15),
    'classifier__fit_intercept':[True,False],
    'classifier__positive':[True,False]
}
model=GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    )

model.fit(x_train, y_train)


models_dir = '../files/models'
os.makedirs(models_dir, exist_ok=True)

# Nombre del archivo comprimido
compressed_model_path = "files/models/model.pkl.gz"


with gzip.open(compressed_model_path, "wb") as file:
    pickle.dump(model, file)


def calculate_and_save_metrics(model, X_train, X_test, y_train, y_test):
    # Hacer predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calcular métricas para el conjunto de entrenamiento
    metrics_train = {
        'type': 'metrics',
        'dataset': 'train',
        'r2': float(r2_score(y_train, y_train_pred)),
        'mse': float(mean_squared_error(y_train, y_train_pred)),
        'mad': float(median_absolute_error(y_train, y_train_pred))
    }

    # Calcular métricas para el conjunto de prueba
    metrics_test = {
        'type': 'metrics',
        'dataset': 'test',
        'r2': float(r2_score(y_test, y_test_pred)),
        'mse': float(mean_squared_error(y_test, y_test_pred)),
        'mad': float(median_absolute_error(y_test, y_test_pred)),
    }

    # Crear carpeta si no existe
    output_dir = 'files/output'
    os.makedirs(output_dir, exist_ok=True)

    # Guardar las métricas en un archivo JSON
    output_path = os.path.join(output_dir, 'metrics.json')
    with open(output_path, 'w') as f:  # Usar 'w' para comenzar con un archivo limpio
        f.write(json.dumps(metrics_train) + '\n')
        f.write(json.dumps(metrics_test) + '\n')

calculate_and_save_metrics(model, x_train, x_test, y_train, y_test)


# %%

