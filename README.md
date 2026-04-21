# Laboratorio Machine Learning Interactivo

Visualizador interactivo de conceptos básicos de **Machine Learning** construido con **Python**, **Pygame**, **NumPy** y **scikit-learn**.

<span><img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue"/></span>

El objetivo del programa es ofrecer un entorno simple y visual para explorar cómo cambian las fronteras de decisión de distintos modelos de clasificación al modificar el conjunto de datos, agregar muestras manualmente y variar algunos parámetros del modelo.

Este proyecto fue desarrollado por el autor **con apoyo de ChatGPT** como asistente de análisis, diseño y programación.

---

## ¿Qué hace este programa?

El programa genera distintos **datasets sintéticos bidimensionales** y permite entrenar sobre ellos varios clasificadores clásicos. Luego representa en pantalla la **frontera de decisión** del modelo, los puntos de cada clase y distintas ayudas visuales para entender qué está haciendo el algoritmo.

Además de los modelos de `scikit-learn`, el proyecto incluye un pequeño **modo de entrenamiento manual de perceptrón**, pensado con fines didácticos, para mostrar paso a paso cómo se ajusta una frontera lineal simple.

En otras palabras, no se trata solamente de “usar modelos”, sino de **verlos trabajar**.

<span><img src="https://github.com/VintaBytes/laboratorio-ml-interactivo/blob/main/images/5.png"  width="600px"/></span>

---

## Características principales

- Generación de datasets sintéticos sin depender de archivos externos.
- Visualización en tiempo real de la frontera de decisión.
- Soporte para varios modelos de clasificación.
- Cambio rápido entre datasets y modelos desde el teclado.
- Agregado manual de nuevos puntos con el mouse.
- Visualización del comportamiento local del clasificador bajo el cursor.
- Modo especial de entrenamiento manual de un perceptrón.
- Interfaz simple, directa y apropiada para uso educativo.
- Ventana redimensionable, con tamaño de fuente ajustado automáticamente.

<span><img src="https://github.com/VintaBytes/laboratorio-ml-interactivo/blob/main/images/1.png"  width="600px"/></span>

---

## Modelos incluidos

El archivo `ml_utils.py` encapsula el entrenamiento de cuatro modelos distintos:

- **Regresión logística** (`LogisticRegression`)
- **SVM** (`SVC`)
- **K-Nearest Neighbors** (`KNeighborsClassifier`)
- **Árbol de decisión** (`DecisionTreeClassifier`)

Cada uno de ellos se entrena sobre un conjunto de datos bidimensional, lo que permite representar el resultado en pantalla de manera clara y comprensible.

### Parámetro ajustable por modelo

La tecla `+` y la tecla `-` permiten aumentar o disminuir un parámetro general del modelo actual:

- En **KNN**, modifica la cantidad de vecinos `k`.
- En **SVM**, modifica el valor de `C`.
- En **árbol de decisión**, modifica la profundidad máxima `max_depth`.
- En **regresión logística**, en esta versión el parámetro visible cambia, pero no se utiliza efectivamente para reconfigurar el modelo.

<span><img src="https://github.com/VintaBytes/laboratorio-ml-interactivo/blob/main/images/6.png"  width="600px"/></span>

---

## Datasets disponibles

El programa puede generar cinco tipos de datasets sintéticos:

- **blobs**
- **moons**
- **circles**
- **classification**

Estos conjuntos se generan mediante funciones de `sklearn.datasets`, lo que vuelve al proyecto muy cómodo para clases, demostraciones o experimentación rápida.

- **Spiral**
- **Random**

Estos conjuntos se generan "matemáticamente", y son interesantes para probar los límites de algunos modelos cuando los datos están agrupados en formas intrincadas, o directamente no hay patrones claros.

<span><img src="https://github.com/VintaBytes/laboratorio-ml-interactivo/blob/main/images/8.png"  width="600px"/></span>

---

## Modos de visualización del mapa

La aplicación puede mostrar tres tipos de mapa para la región de decisión:

### 1. Modo normal
Muestra el plano coloreado según la clase predicha por el modelo.

### 2. Modo probabilidad
Cuando el modelo lo permite, utiliza `predict_proba()` para mostrar un gradiente de color que refleja la confianza de la predicción.

### 3. Modo incertidumbre
Representa visualmente las zonas donde la predicción es menos segura, es decir, donde la probabilidad está más cerca de 0.5.

Este punto es especialmente útil para explicar por qué algunas regiones del espacio son más ambiguas que otras.

<span><img src="https://github.com/VintaBytes/laboratorio-ml-interactivo/blob/main/images/7.png"  width="600px"/></span>

---

## Interacciones disponibles

## Controles por teclado

- `D` : cambia el dataset actual.
- `R` : regenera el dataset.
- `M` : cambia el modelo actual.
- `B` : cambia el tipo de mapa.
- `V` : muestra u oculta ayudas visuales del modelo.
- `+` : aumenta el parámetro del modelo.
- `-` : disminuye el parámetro del modelo.
- `H` : muestra u oculta la ayuda en pantalla.
- `T` : activa o desactiva el modo de entrenamiento manual del perceptrón.
- `SPACE` : ejecuta un paso de entrenamiento del perceptrón.
- `A` : activa o desactiva el entrenamiento automático del perceptrón.

## Controles con mouse

- **Click izquierdo**: agrega un nuevo punto de la clase azul.
- **Click derecho**: agrega un nuevo punto de la clase roja.
- **Mover el mouse**: muestra la predicción local bajo el cursor.

Según el modelo seleccionado, también se pueden mostrar ayudas adicionales:

- En **KNN**, se dibujan los vecinos más cercanos al cursor y el radio que los contiene.
- En **SVM**, se pueden resaltar los **support vectors**.

<span><img src="https://github.com/VintaBytes/laboratorio-ml-interactivo/blob/main/images/2.png"  width="600px"/></span>

---

## Modo perceptrón

Uno de los aspectos más interesantes del proyecto es el modo especial de entrenamiento manual.

Cuando se activa con la tecla `T`, el programa deja de enfocarse en los modelos de `scikit-learn` y comienza a mostrar una simulación didáctica de un **perceptrón binario** implementado a mano.

En este modo:

- se inicializan pesos aleatorios,
- se recorre el dataset punto por punto,
- se calcula la salida con el signo del producto interno,
- y, si la clasificación es incorrecta, se actualizan pesos y bias mediante una regla simple de aprendizaje.

Esto permite observar visualmente:

- qué punto está siendo procesado,
- cómo cambia la recta del perceptrón,
- y cómo evoluciona el entrenamiento paso a paso.

Esta parte del programa tiene solamente valor pedagógico para enseñanza introductoria de clasificación lineal.

---

## Estructura del proyecto

### `main.py`
Es el archivo principal de la aplicación. Allí se concentran:

- la inicialización de Pygame,
- el estado global del programa,
- la gestión de eventos de teclado y mouse,
- la conversión entre coordenadas del mundo y coordenadas de pantalla,
- el dibujo del mapa de decisión,
- el renderizado de puntos, HUD y ayuda,
- y la lógica del perceptrón manual.

### `ml_utils.py`
Este módulo agrupa funciones auxiliares vinculadas a machine learning:

- `generar_dataset(...)`
- `entrenar_modelo(...)`
- `calcular_frontera(...)`

Separar estas funciones del archivo principal mejora bastante la legibilidad del proyecto y hace más fácil su ampliación futura.

<span><img src="https://github.com/VintaBytes/laboratorio-ml-interactivo/blob/main/images/4.png"  width="600px"/></span>

---

## Dependencias

Para ejecutar el programa se necesitan, como mínimo, estas bibliotecas:

- `pygame`
- `numpy`
- `scikit-learn`

Instalación sugerida:

```bash
pip install pygame numpy scikit-learn
```

---

## Ejecución

Una vez instaladas las dependencias:

```bash
python main.py
```

El programa abrirá una ventana gráfica redimensionable y mostrará automáticamente un dataset inicial junto con su modelo actual.

---

## Cómo usarlo

Una forma simple de explorar el programa es la siguiente:

1. Ejecutar la aplicación.
2. Cambiar entre datasets con `D`.
3. Cambiar entre modelos con `M`.
4. Observar cómo cambia la frontera de decisión.
5. Ajustar parámetros con `+` y `-`.
6. Cambiar el tipo de mapa con `B`.
7. Agregar puntos manualmente con clicks del mouse.
8. Activar `V` para mostrar vecinos KNN o support vectors en SVM.
9. Entrar en modo perceptrón con `T` y avanzar paso a paso con `SPACE`.

Esta dinámica vuelve al programa especialmente útil para:

- clases introductorias de Machine Learning,
- demostraciones visuales,
- análisis de clasificadores en 2D,
- práctica de conceptos como frontera de decisión, probabilidad e incertidumbre.

---

## Aspectos didácticos destacados

Este proyecto resulta interesante porque combina tres niveles de aprendizaje:

1. **Uso de modelos ya implementados** en `scikit-learn`.
2. **Visualización geométrica** de lo que el modelo aprende.
3. **Implementación manual** de un perceptrón sencillo para comprender la lógica de entrenamiento.

Ese cruce entre programación, visualización y teoría hace que el programa sea adecuado para un “laboratorio” de aprendizaje automático orientado a principiantes.

<span><img src="https://github.com/VintaBytes/laboratorio-ml-interactivo/blob/main/images/3.png"  width="600px"/></span>

---

## Posibles mejoras futuras

Algunas extensiones interesantes que pienso explorar para continuar el proyecto incluyen:

- agregar una leyenda visual más desarrollada,
- permitir elegir parámetros específicos por modelo,
- incorporar más algoritmos,
- mostrar matrices de confusión o métricas adicionales,
- guardar y cargar datasets manuales,
- agregar una interfaz de panel lateral con botones y sliders,
- incluir una pequeña explicación textual del modelo seleccionado.

---

## Licencia 

Se permite:

- usar el código,
- copiarlo,
- compartirlo,
- modificarlo,
- y adaptarlo,

siempre que:

1. se cite claramente al autor original,
2. se conserve la referencia al proyecto original,
3. no se utilice con fines comerciales sin autorización expresa,
4. y se indique si se realizaron modificaciones.

En el archivo `LICENSE` se puede ver la redacción completa.

---

## Créditos

Proyecto desarrollado por el autor para exploración y enseñanza de conceptos de Machine Learning, con asistencia de **ChatGPT** en tareas de análisis, diseño y programación.
