## Ejemplo de Respuesta Educativa

```json
{
  "success": true,
  "# Servicio de Evaluación de Escritura para Alfabetización"
}
```
## Diseñado Específicamente Para
- **Jóvenes y adultos en proceso de alfabetización**
- **Evaluación de escritura a mano de letras del alfabeto**
- **Retroalimentación educativa personalizada**
- **Análisis detallado de técnica de escritura**

## Instalación de Dependencias

Crea un archivo `requirements.txt`:

```
Flask==3.0.0
Werkzeug==3.0.1
opencv-python==4.8.1.78
Pillow==10.1.0
scikit-image==0.22.0
numpy==1.24.3
scipy==1.11.4
python-multipart==0.0.6
```

Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Ejecución del Servicio

```bash
python app.py
```

El servicio estará disponible en: `http://localhost:5001`

## 🔧 Algoritmos Especializados para Escritura

### 1. **Filtros Direccionales para Trazos**
- **Trazos Verticales**: Detecta líneas como en las letras I, l, t, d
- **Trazos Horizontales**: Analiza líneas como en H, F, E, T
- **Trazos Diagonales**: Evalúa líneas inclinadas en A, V, W, X
- **Detección de Curvas**: Analiza formas redondeadas en O, C, S, P

### 2. **Preprocesamiento Optimizado**
- **Binarización Adaptativa**: Maneja diferentes tipos de papel e iluminación
- **Centrado Automático**: Localiza y centra la letra automáticamente
- **Limpieza de Ruido**: Elimina manchas y artefactos del escaneo
- **Normalización de Grosor**: Ajusta diferencias en el grosor del trazo

### 3. **Análisis de Calidad Específico**
- **Consistencia de Trazos**: Evalúa uniformidad en líneas rectas
- **Proporciones**: Compara tamaños relativos de la letra
- **Conectores**: Analiza uniones entre diferentes partes de la letra
- **Densidad de Trazos**: Evalúa distribución del "peso" de la letra

## Ejemplos de Uso

### 1. **Usando cURL**

```bash
curl -X POST http://localhost:5000/image/compare \
  -F "original_img=@imagen_original.jpg" \
  -F "compare_img=@imagen_estudiante.jpg"
```

### 2. **Usando Python con requests**

```python
import requests

url = "http://localhost:5000/image/compare"

files = {
    'original_img': ('original.jpg', open('imagen_original.jpg', 'rb')),
    'compare_img': ('compare.jpg', open('imagen_estudiante.jpg', 'rb'))
}

response = requests.post(url, files=files)
result = response.json()

print(f"Precisión: {result['precision']}")
print(f"Porcentaje: {result['precision'] * 100:.2f}%")
```

### 3. **Usando JavaScript (Frontend)**

```javascript
const formData = new FormData();
formData.append('original_img', originalFileInput.files[0]);
formData.append('compare_img', compareFileInput.files[0]);

fetch('http://localhost:5000/image/compare', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Precisión:', data.precision);
    console.log('Porcentaje:', (data.precision * 100).toFixed(2) + '%');
})
.catch(error => console.error('Error:', error));
```

## Ejemplo de Respuesta

```json
{
  "success": true,
  "precision": 0.8542,
  "message": "Precisión calculada: 85.42%",
  "algorithm_details": {
    "method": "Convolución con filtros Sobel, Gaussiano y Laplaciano",
    "metrics_used": [
      "Detección de bordes horizontales y verticales",
      "Suavizado Gaussiano", 
      "Detección de bordes Laplaciano",
      "Índice de Similitud Estructural (SSIM)",
      "Similitud pixel a pixel"
    ]
  },
  "feature_analysis": {
    "feature_similarities": {
      "edges_x": 0.8234,
      "edges_y": 0.8156, 
      "smoothed": 0.8891,
      "edges_laplacian": 0.8012,
      "edge_magnitude": 0.8345
    },
    "ssim_score": 0.8723,
    "pixel_similarity": 0.7892
  }
}
```

## Cómo Funciona la Evaluación

1. **Preprocesamiento**:
   - Redimensiona ambas imágenes a 256x256 píxeles
   - Convierte a escala de grises
   - Normaliza valores entre 0 y 1

2. **Aplicación de Filtros de Convolución**:
   - Sobel X/Y para detección de bordes
   - Gaussiano para suavizado
   - Laplaciano para detección de características

3. **Cálculo de Similitud**:
   - Correlación entre características extraídas
   - SSIM (Structural Similarity Index)
   - Similitud pixel a pixel

4. **Puntuación Final**:
   - Combina todas las métricas con pesos específicos
   - Resultado entre 0.0 (sin similitud) y 1.0 (idénticas)

## Estructura del Proyecto

```
proyecto/
├── app.py              # Servicio principal
├── requirements.txt    # Dependencias
├── test_client.py     # Cliente de prueba
├── temp_uploads/      # Directorio temporal (creado automáticamente)
└── README.md          # Esta documentación
```

## Notas Importantes

- **Formatos soportados**: PNG, JPG, JPEG, GIF, BMP
- **Tamaño máximo**: 16MB por archivo
- **Los archivos temporales se eliminan automáticamente**
- **El servicio usa algoritmos de convolución reales para la comparación**
- **La precisión se calcula combinando múltiples métricas**