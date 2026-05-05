# Clasificador de Manzanas y Jitomates (CNN)
Este proyecto usa una Red Neuronal Convolucional para diferenciar frutas.

## Cómo usar
1. Instala las dependencias: `pip install torch torchvision`
2. Para entrenar: ejecuta `python cnn_frutas.py`
3. Para probar una imagen: ejecuta `python predecir_fruta.py`

**Nota:** El dataset no está incluido por su peso, pero puedes usar tus propias imágenes en la carpeta `train/test`.

## Requisitos
Para ejecutar este proyecto, necesitas tener instalado Python y las siguientes librerías:
* **PyTorch** (torch)
* **Torchvision**
* **Pillow** (para el manejo de imágenes `PIL`)
