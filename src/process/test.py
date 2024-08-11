# %%
import cv2
import os

images = os.listdir("/Users/nahueltabasso/Documents/Python/yolo_fine_tunning/data/dataset-240728/dataset-240728/images/train")
print
for i in images:
    img = cv2.imread(os.path.join("/Users/nahueltabasso/Documents/Python/yolo_fine_tunning/data/dataset-240728/dataset-240728/images/train", i))
    # if img.shape != (640, 640, 3):
    print(img.shape)
    
    
    
# %%
# %%
import cv2
import matplotlib.pyplot as plt

path = "/Users/nahueltabasso/Documents/Python/yolo_fine_tunning/data/fo-dataset-240728/images/test"

images = os.listdir(path)
len(images)

# %%
num_images = 36  # o 35

# Configuración de la cuadrícula
images_per_row = 6
num_rows = (num_images // images_per_row) + (1 if num_images % images_per_row != 0 else 0)

num_rows
# %%
# Crear una figura grande para acomodar todas las imágenes
fig, axes = plt.subplots(num_rows, images_per_row, figsize=(20, num_rows * 3))

# Aplanar la matriz de ejes para un acceso más fácil
axes = axes.flatten()

for i in range(num_images):
    # Cargar la imagen
    img = cv2.imread(os.path.join(path, images[i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Mostrar la imagen
    axes[i].imshow(img)
    axes[i].axis('off')  # Ocultar los ejes
    axes[i].set_title(images[i], fontsize=12)


# Si hay espacios vacíos en la cuadrícula, desactivar los ejes
for i in range(num_images, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# %%
import numpy as np


def plot_confusion_matrix(confusion_matrix, class_labels):
    """
    Genera una imagen de la matriz de confusión utilizando Matplotlib.
    
    Args:
        confusion_matrix (numpy.ndarray): Matriz de confusión.
        class_labels (list): Lista de etiquetas de clase.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Crear el heatmap
    im = ax.imshow(confusion_matrix, cmap='Blues')
    
    # Agregar etiquetas y título
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel('Predictions')
    ax.set_ylabel('Real Classes')
    ax.set_title('Confusion Matrix')
    
    # Agregar valores numéricos en la matriz
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            ax.text(j, i, int(confusion_matrix[i, j]), ha="center", va="center", color="white")
    
    fig.colorbar(im, ax=ax)
    plt.savefig('confusion_matrix.png')

# Ejemplo de uso
class_labels = ['Credit Card', 'Object']
confusion_matrix = np.array([[37, 0], [1, 0]])
plot_confusion_matrix(confusion_matrix, class_labels)
