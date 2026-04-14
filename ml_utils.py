# --------------------------------------------------------------
# Importamos librerias
# --------------------------------------------------------------
import numpy as np
from sklearn.datasets import (
    make_blobs, 
    make_moons, 
    make_circles,
    make_classification)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# --------------------------------------------------------------
# Generamos el set de datos
# --------------------------------------------------------------
def generar_dataset(tipo="blobs", n=60, ruido=0.2):
    if tipo == "blobs":
        X, y = make_blobs(n_samples=n, centers=2, random_state=0)
    elif tipo == "moons":
        X, y = make_moons(n_samples=n, noise=ruido)
    elif tipo == "circles":
        X, y = make_circles(n_samples=n, noise=ruido, factor=0.5)
    elif tipo == "classification":
        X, y = make_classification(
            n_samples=n,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            class_sep=0.8,
            random_state=0
        )
    return X, y

# --------------------------------------------------------------
# Entrena el modelo elegido con el nro de parámetros
# --------------------------------------------------------------
def entrenar_modelo(tipo, X, y, parametro=5):
    if tipo == "logistic":
        modelo = LogisticRegression()
    elif tipo == "knn":
        modelo = KNeighborsClassifier(n_neighbors=parametro)
    elif tipo == "svm":
        modelo = SVC(C=parametro, probability=True)
    elif tipo == "tree":
        modelo = DecisionTreeClassifier(max_depth=parametro)
    else:
        return None

    modelo.fit(X, y)

    return modelo

# --------------------------------------------------------------
# Calcula la frontera
# --------------------------------------------------------------
def calcular_frontera(modelo, xmin, xmax, ymin, ymax, resolucion=200):
    xs = np.linspace(xmin, xmax, resolucion)
    ys = np.linspace(ymin, ymax, resolucion)
    xx, yy = np.meshgrid(xs, ys)
    grid_puntos = np.c_[xx.ravel(), yy.ravel()]
    predicciones = modelo.predict(grid_puntos)
    grid = predicciones.reshape(xx.shape)
    return grid, xs, ys


