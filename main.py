import cv2
import numpy as np

# Cargar el clasificador pre-entrenado para la detección de objetos (por ejemplo, personas)
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar la cámara
cap = cv2.VideoCapture(0)

while True:
    # Capturar fotograma por fotograma
    ret, frame = cap.read()

    # Convertir el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar objetos en el fotograma
    objects = classifier.detectMultiScale(gray, 1.3, 5)

    # Dibujar un cuadro verde alrededor de los objetos detectados
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostrar el fotograma con los objetos detectados
    cv2.imshow('Object Detection', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
