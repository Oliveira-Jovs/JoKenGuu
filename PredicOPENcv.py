import cv2
from ultralytics import YOLO
import torch

model = YOLO(r"C:\Users\oliveira\Desktop\Projetos_Pycharm\FAP_YOLO_APRESENTACAO\runs\detect\train\weights\best.pt")
device = torch.device("cpu")
model.to(device)

results = model.predict(
    r"C:\Users\oliveira\Desktop\Projetos_Pycharm\FAP_YOLO_APRESENTACAO\Jokenpo\valid\images\0037_png.rf.6231d8fc3fe6083092067e96ac3cd715.jpg")

# Acessando os resultados (a lista de predições)
predictions = results[0]  # A primeira (e provavelmente única) previsão

# Carregando a imagem original
image = cv2.imread(
    r"C:\Users\oliveira\Desktop\Projetos_Pycharm\FAP_YOLO_APRESENTACAO\Jokenpo\valid\images\0037_png.rf.6231d8fc3fe6083092067e96ac3cd715.jpg")

# Desenhando as caixas delimitadoras na imagem
for result in predictions.boxes:  # 'boxes' contém as coordenadas das caixas
    x1, y1, x2, y2 = result.xyxy[0]  # As coordenadas da caixa (x1, y1, x2, y2)
    conf = result.conf[0]  # Confiança da predição
    cls = result.cls[0]  # Classe da predição

    # Convertendo as coordenadas para inteiros
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # Desenhando o retângulo
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Verde, espessura 2

    # Colocando o nome da classe na imagem
    label = f"{model.names[int(cls)]}: {conf:.2f}"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Exibindo a imagem com as caixas delimitadoras
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
