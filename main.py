# --------------------------------------------------------------
# Importamos librerias
# --------------------------------------------------------------
import pygame
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ml_utils import (
    generar_dataset, 
    entrenar_modelo, 
    calcular_frontera, 
)

# ANCHO = 1280
# ALTO = 720
COLOR_FONDO = (20, 20, 30)
AZUL = (80, 150, 255)
ROJO = (255, 100, 100)
COLOR_LINEA = (255,255,255)

# --------------------------------------------------------------
# Variables para el "modo entrenamiento", etc
# --------------------------------------------------------------
modo_entrenamiento = False
pesos = None
bias = 0
indice_entrenamiento = 0
learning_rate = 0.1
auto_entrenar = False
mostrar_ayuda = False
mostrar_vecinos = False
accuracy = 0
modo_mapa = 0   # 0 normal, 1 probabilidad, 2 incertidumbre
modelos = ["logistic", "svm", "knn", "tree"]
datasets = ["blobs", "moons", "circles", "classification"]

# --------------------------------------------------------------
# Ventana gráfica
# --------------------------------------------------------------
pygame.init()
info = pygame.display.Info()
ANCHO = info.current_w
ALTO = info.current_h
pantalla = pygame.display.set_mode((ANCHO, ALTO), pygame.RESIZABLE)
pygame.display.set_caption("ML Visualizer")
clock = pygame.time.Clock()

# Elijo la fuente "dejavusansmono" si existe.
ruta_fuente = pygame.font.match_font("dejavusansmono")
if ruta_fuente is None:
    ruta_fuente = pygame.font.match_font("liberationmono")

def actualizar_fuente():
    global font, tamaño_fuente
    tamaño_fuente = max(12, int(ALTO * 0.03))
    tamaño_fuente = min(22, tamaño_fuente )
    font = pygame.font.Font(ruta_fuente, tamaño_fuente)

actualizar_fuente()

# --------------------------------------------------------------
# Estado del programa
# --------------------------------------------------------------
dataset_actual = "moons"
modelo_actual = "knn"
parametro_modelo = 5

# --------------------------------------------------------------
# Funciones que generan el dataset y entrenan el modelo.
# --------------------------------------------------------------
def generar_nuevo_dataset():
    global X, y
    global xmin, xmax, ymin, ymax

    X, y = generar_dataset(dataset_actual)

    xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
    ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1

    actualizar_split()


# --------------------------------------------------------------
# Reentrena el modelo si modifique datos.
# --------------------------------------------------------------
def reentrenar():
    global modelo, grid, xs, ys
    global train_acc, test_acc, prob_map

    # entrenar modelo SOLO con entrenamiento
    modelo = entrenar_modelo(modelo_actual, X_train, y_train, parametro_modelo)

    # resolución frontera
    if modelo_actual == "tree":
        resolucion = 240
    else:
        resolucion = 120

    grid, xs, ys = calcular_frontera(
        modelo, xmin, xmax, ymin, ymax, resolucion
    )

    # accuracy entrenamiento
    pred_train = modelo.predict(X_train)
    train_acc = accuracy_score(y_train, pred_train)

    # accuracy test
    pred_test = modelo.predict(X_test)
    test_acc = accuracy_score(y_test, pred_test)

    # mapa probabilidades
    prob_map = None
    if hasattr(modelo, "predict_proba"):
        xx, yy = np.meshgrid(xs, ys)
        puntos = np.c_[xx.ravel(), yy.ravel()]
        probs = modelo.predict_proba(puntos)[:, 1]
        prob_map = probs.reshape(xx.shape)

# --------------------------------------------------------------
def reiniciar_dataset():
    generar_nuevo_dataset()
    reentrenar()

# --------------------------------------------------------------
# Conversión coordenadas
# --------------------------------------------------------------
def mundo_a_pantalla(x,y):
    px = int((x-xmin)/(xmax-xmin)*ANCHO)
    py = int(ALTO-(y-ymin)/(ymax-ymin)*ALTO)
    return px,py

def pantalla_a_mundo(px, py):
    x = xmin + (px / ANCHO) * (xmax - xmin)
    y = ymin + ((ALTO - py) / ALTO) * (ymax - ymin)
    return x, y

# --------------------------------------------------------------
# Dibujar linea de decisión
# --------------------------------------------------------------
def dibujar_linea_decision():
    if modelo_actual not in ["logistic","svm"]:
        return
    try:
        w = modelo.coef_[0]
        b = modelo.intercept_[0]
        x1 = xmin
        x2 = xmax
        y1 = -(w[0]*x1 + b)/w[1]
        y2 = -(w[0]*x2 + b)/w[1]
        p1 = mundo_a_pantalla(x1,y1)
        p2 = mundo_a_pantalla(x2,y2)
        pygame.draw.line(pantalla, COLOR_LINEA, p1, p2, 3)
    except:
        pass


# --------------------------------------------------------------
# Dibujar Support Vectors en SVM
# --------------------------------------------------------------
def dibujar_support_vectors():
    if modelo_actual != "svm":
        return
    if not mostrar_vecinos:
        return
    if not hasattr(modelo, "support_vectors_"):
        return

    for p in modelo.support_vectors_:
        px, py = mundo_a_pantalla(p[0], p[1])
        pygame.draw.circle(
            pantalla,
            (255,255,120),
            (px, py),
            12,
            2
        )

# --------------------------------------------------------------
# Muestra "vecinos" en knn
# --------------------------------------------------------------
def dibujar_vecinos_knn():
    if modelo_actual != "knn":
        return
    if not mostrar_vecinos:
        return
    
    px, py = pygame.mouse.get_pos()
    x_mundo, y_mundo = pantalla_a_mundo(px, py)
    punto = np.array([x_mundo, y_mundo])
    distancias = np.linalg.norm(X - punto, axis=1)
    indices = np.argsort(distancias)[:parametro_modelo]
    radio = distancias[indices[-1]]

    for idx in indices:
        vecino = X[idx]
        vx, vy = mundo_a_pantalla(vecino[0], vecino[1])
        pygame.draw.line(pantalla, (255,255,255), (px,py), (vx,vy), 1)
        pygame.draw.circle(pantalla, (255,255,255), (vx,vy), 10, 2)

    pygame.draw.circle(pantalla, (255,255,255), (px,py), 5)
    radio_px = int(radio / (xmax - xmin) * ANCHO)

    pygame.draw.circle(
        pantalla,
        (200,200,200),
        (px,py),
        radio_px,
        1
    )

# --------------------------------------------------------------
# Inicializar el perceptrón
# --------------------------------------------------------------
def iniciar_entrenamiento():
    global pesos, bias, indice_entrenamiento
    pesos = np.random.randn(2)
    bias = 0
    indice_entrenamiento = 0

# ------------------------------
# Paso de entrenamiento
# ------------------------------
def paso_entrenamiento():
    global pesos, bias, indice_entrenamiento
    x = X[indice_entrenamiento]
    y_real = 1 if y[indice_entrenamiento] == 1 else -1
    salida = np.sign(np.dot(pesos, x) + bias)

    if salida != y_real:
        pesos = pesos + learning_rate * y_real * x
        bias = bias + learning_rate * y_real

    indice_entrenamiento = (indice_entrenamiento + 1) % len(X)

# ------------------------------
# Dibujar la línea del perceptrón
# ------------------------------
def dibujar_linea_perceptron():
    if pesos is None:
        return

    w1, w2 = pesos
    x1 = xmin
    x2 = xmax
    y1 = -(w1*x1 + bias)/w2
    y2 = -(w1*x2 + bias)/w2
    p1 = mundo_a_pantalla(x1, y1)
    p2 = mundo_a_pantalla(x2, y2)
    pygame.draw.line(pantalla, (255,255,0), p1, p2, 3)

# ------------------------------
# Mostrar el punto que está 
# entrenando el perceptrón.
# ------------------------------
def dibujar_punto_entrenamiento():
    if not modo_entrenamiento:
        return
    if pesos is None:
        return

    x = X[indice_entrenamiento]
    px, py = mundo_a_pantalla(x[0], x[1])
    pygame.draw.circle(pantalla, (255,255,0), (px,py), 12, 2)


# --------------------------------------------------------------
# Dibuja el "cursor" del mouse del color predicho
# --------------------------------------------------------------
def dibujar_prediccion_mouse():
    px, py = pygame.mouse.get_pos()
    x_mundo, y_mundo = pantalla_a_mundo(px, py)
    punto = np.array([[x_mundo, y_mundo]])
    # if not mostrar_vecinos:
    #     return
    try:
        pred = modelo.predict(punto)[0]
    except:
        return

    color = AZUL if pred == 0 else ROJO
    pygame.draw.circle(pantalla, color, (px, py), 8, 2)

    # calcular probabilidad
    prob_texto = ""
    if hasattr(modelo, "predict_proba"):
        probs = modelo.predict_proba(punto)[0]
        prob = probs[pred]
        prob_texto = f"{prob:.2f}"

    # dibujar texto
    if prob_texto:
        txt = font.render(prob_texto, True, (255,255,255))
        pantalla.blit(txt, (px + 10, py + 10))


# --------------------------------------------------------------
# Pantalla ayuda
# --------------------------------------------------------------
def dibujar_ayuda():
    lineas = [
        "TECLADO:",
        "[H] Mostrar / ocultar ayuda",
        "",
        "DATASETS:",
        "[D] Bloobs | Moons | Circles | Classification",
        "[R] Regenerar dataset",
        "",
        "MODELOS:",
        "[M] LR | SVM | KNN | Decision Tree",
        "[V] Mostrar / ocultar vecinos KNN",
        "[B] Cambiar tipo de mapa",
        "[+] Aumentar parametro",
        "[-] Disminuir parametro",
        "",
        "PERCEPTRON (MODO ENTRENAMIENTO):",
        "    [T] > Activar / desactivar modo",
        "[SPACE] > Paso de entrenamiento",
        "    [A] > Entrenamiento automatico",
        "",
        "MOUSE:",
        " [Izq.] Agregar punto azul",
        " [Der.] Agregar punto rojo",
        "[Mover] Ver vecinos KNN"
    ]

    y_pos = tamaño_fuente * 4

    for linea in lineas:
        if linea.isupper():
            color = (255,220,120)
        else:
            color = (255,255,255)
        txt = font.render(linea,True,color)
        pantalla.blit(txt,(50,y_pos))
        y_pos += tamaño_fuente + 2

 
# --------------------------------------------------------------
# Dibuja datos del modelo actual en pantalla
# --------------------------------------------------------------
def dibujar_hud():
    modo_texto = ["Normal","Probabilidad","Incertidumbre"][modo_mapa]
    color = (255,220,120) if modo_entrenamiento else (255,255,255)
    linea1 = [
        f"   Dataset: {dataset_actual:14} ",
        f" Modelo: {modelo_actual:9} "
        f" Mapa: {modo_texto:10}"
    ]
    if modo_entrenamiento:
        linea2 = [
            "      Modo: PERCEPTRON",
            f" Paso: {indice_entrenamiento:<8}",
            f"  Learnig Rate: {learning_rate:.2f}"
        ]
    else:
        linea2 = [
            f"Parametros: {parametro_modelo:2}{"":12}",
            f"   Train: {train_acc:.3f}{"":4}"
            f"  Test: {test_acc:.3f}"
        ]
    texto1 = "   ".join(linea1)
    texto2 = "   ".join(linea2)
    surf1 = font.render(texto1, True, color)
    surf2 = font.render(texto2, True, color)
    pantalla.blit(surf1,(20,6))
    pantalla.blit(surf2,(20,6 + tamaño_fuente + 4))


# --------------------------------------------------------------
# Recalcula train/test a partir del dataset actual
# --------------------------------------------------------------
def actualizar_split():
    global X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

# --------------------------------------------------------------
# Inicialización del modelo y loop principal
# --------------------------------------------------------------
reiniciar_dataset()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            # Cambio de dataset
            if event.key == pygame.K_d:
                i = datasets.index(dataset_actual)
                dataset_actual = datasets[(i + 1) % len(datasets)]
                generar_nuevo_dataset()
                reentrenar()

            # Regenero el dataset
            elif event.key == pygame.K_r:
                    generar_nuevo_dataset()
                    reentrenar()                
            
            # Cambio de modelo
            elif event.key == pygame.K_m:
                i = modelos.index(modelo_actual)
                modelo_actual = modelos[(i + 1) % len(modelos)]
                reentrenar()

            # Cambio modo de representacion
            elif event.key == pygame.K_b:
                modo_mapa = (modo_mapa + 1) % 3

            elif event.key == pygame.K_v:
                mostrar_vecinos = not mostrar_vecinos
            elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                parametro_modelo += 1
                reentrenar()
            elif event.key == pygame.K_MINUS:
                parametro_modelo = max(1,parametro_modelo-1)
                reentrenar()
            elif event.key == pygame.K_h:
                mostrar_ayuda = not mostrar_ayuda

            # Teclas Modo Entrenamiento
            elif event.key == pygame.K_t:
                modo_entrenamiento = not modo_entrenamiento
                if modo_entrenamiento:
                    iniciar_entrenamiento()
            elif event.key == pygame.K_SPACE:
                if modo_entrenamiento:
                    paso_entrenamiento()
            elif event.key == pygame.K_a:
                auto_entrenar = not auto_entrenar

        # Controles via mouse
        if event.type == pygame.MOUSEBUTTONDOWN:
            px, py = pygame.mouse.get_pos()
            x, y_mundo = pantalla_a_mundo(px, py)

            if event.button == 1:   # click izquierdo
                X = np.vstack([X, [x, y_mundo]])
                y = np.append(y, 0)
                actualizar_split()
                reentrenar()

            elif event.button == 3: # click derecho
                X = np.vstack([X, [x, y_mundo]])
                y = np.append(y, 1)
                actualizar_split()
                reentrenar()

                # Evento cambio tamaño de ventana
                if event.type == pygame.VIDEORESIZE:
                    ANCHO, ALTO = event.w, event.h
                    actualizar_fuente()

    # ------------------------------
    # Dibujar frontera
    # ------------------------------
    paso_x = ANCHO / len(xs)
    paso_y = ALTO / len(ys)

    for j in range(len(ys)):
        for i in range(len(xs)):
            clase = grid[j][i]
            p = prob_map[j][i]

            if modo_mapa == 0 or prob_map is None:
                clase = grid[j][i]
                color = (40,70,140) if clase==0 else (140,40,40)

            elif modo_mapa == 1:
                p = prob_map[j][i]
                r = int(40 + p * 180)
                b = int(40 + (1 - p) * 180)
                color = (r,40,b)

            elif modo_mapa == 2:
                p = prob_map[j][i]
                incertidumbre = 1 - abs(p - 0.5) * 2
                v = int(80 + incertidumbre * 175)
                color = (v,v,v)

            rect = pygame.Rect(
                i * paso_x,
                ALTO - (j + 1) * paso_y,
                paso_x + 1,
                paso_y + 1
            )

            pygame.draw.rect(pantalla, color, rect)

    # ------------------------------
    # Dibujar puntos
    # ------------------------------
    for i in range(len(X)):
        px,py = mundo_a_pantalla(X[i][0],X[i][1])
        color = AZUL if y[i]==0 else ROJO
        pygame.draw.circle(pantalla,color,(px,py),6)

    # ------------------------------
    # Entrenamiento si corresponde
    # ------------------------------
    if modo_entrenamiento:
        dibujar_punto_entrenamiento()
        if auto_entrenar:
            paso_entrenamiento()

    # ------------------------------
    # Dibujar linea decision
    # ------------------------------
    dibujar_linea_decision()
    dibujar_vecinos_knn()
    dibujar_prediccion_mouse()
    dibujar_support_vectors()
    if modo_entrenamiento:
        dibujar_linea_perceptron()

    # ------------------------------
    # Texto estado
    # ------------------------------
    modo_texto = ["Normal","Probabilidad","Incertidumbre"][modo_mapa]
    if modo_entrenamiento:
        texto = (
            f"Mapa:{modo_texto:<12}"
            f"Modo: PERCEPTRON TRAINING  "
            f"Dataset:{dataset_actual:<8}  "
            f"Paso:{indice_entrenamiento:<4}  "
            f"Learning rate:{learning_rate:.2f}"
        )
    else:
        texto = (
            f"Mapa:{modo_texto:<12}"
            f"Dataset:{dataset_actual:<8}  "
            f"Modelo:{modelo_actual:<8}  "
            f"Parametros:{parametro_modelo:<3}  "
            f"Accuracy:{accuracy:.3f}"
        )
    color_hud = (255,220,120) if modo_entrenamiento else (255,255,255)
    superficie = font.render(texto, True, color_hud)
    dibujar_hud()

    # ------------------------------
    # Ayuda
    # ------------------------------
    if mostrar_ayuda:
        dibujar_ayuda()

    pygame.display.flip()
    clock.tick(10)

pygame.quit()