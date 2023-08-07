import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image

# Ejemplo de imagen aleatoria del JSON
json_data = '{"34020010494_e5cb88e1c4_k.jpg1115004":{"fileref":"","size":1115004,"filename":"34020010494_e5cb88e1c4_k.jpg","base64_img_data":"","file_attributes":{},"regions":{"0":{"shape_attributes":{"name":"polygon","all_points_x":[1020,1000,994,1003,1023,1050,1089,1134,1190,1265,1321,1361,1403,1428,1442,1445,1441,1427,1400,1361,1316,1269,1228,1198,1207,1210,1190,1177,1172,1174,1170,1153,1127,1104,1061,1032,1020],"all_points_y":[963,899,841,787,738,700,663,638,621,619,643,672,720,765,800,860,896,942,990,1035,1079,1112,1129,1134,1144,1153,1166,1166,1150,1136,1129,1122,1112,1084,1037,989,963]},"region_attributes":{}}}}}'

# Convertir JSON a diccionario de Python
data = json.loads(json_data)

# Obtener las coordenadas x e y de los puntos de interés
points_x = data["34020010494_e5cb88e1c4_k.jpg1115004"]["regions"]["0"]["shape_attributes"]["all_points_x"]
points_y = data["34020010494_e5cb88e1c4_k.jpg1115004"]["regions"]["0"]["shape_attributes"]["all_points_y"]

# Cargar la imagen para obtener sus dimensiones
image_path = r"C:\Users\Lenovo\Downloads\balloon_dataset (1)\balloon\train\balloon_dataset_tfg1.jpg"  # Reemplaza esto con la ruta correcta a tu imagen
image = Image.open(image_path)
image_width, image_height = image.size

# Crear el gráfico y mostrar los puntos de interés
fig, ax = plt.subplots(figsize=(image_width / 100, image_height / 100))  # Escalar el gráfico para que tenga las mismas dimensiones que la imagen
ax.imshow(image)
ax.add_patch(Polygon(xy=list(zip(points_x, points_y))))
ax.plot(points_x, points_y, 'ro')  # Puntos de interés representados por círculos rojos
plt.title('Puntos de interés de imagen aleatoria en balloon dataset')
plt.show()
