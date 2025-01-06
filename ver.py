from ultralytics import YOLO

modelo = YOLO("C:/Users/oliveira/Desktop/Projetos_Pycharm/FAP_YOLO_APRESENTACAO/runs/detect/train/weights/best.pt")

metrica = modelo.val()

print(metrica.box.map)
print()
print(metrica.box.map50)
print()
print(metrica.box.map75)