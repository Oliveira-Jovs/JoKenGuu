import pandas as pd
pd.set_option('display.max_columns', None)

df = pd.read_csv(r'C:\Users\oliveira\Desktop\Projetos_Pycharm\FAP_YOLO_APRESENTACAO\runs\detect\train\results.csv')

print(df.tail())
