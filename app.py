from flask import Flask, request, jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os
import tempfile
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB máximo

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Crear directorio temporal si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Verificar si el archivo tiene una extensión permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocesar imagen optimizado para escritura a mano
    - Limpiar ruido de fondo
    - Binarizar para resaltar trazos
    - Centrar y redimensionar letra
    - Normalizar grosor de línea
    """
    # Leer imagen
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro bilateral para reducir ruido manteniendo bordes
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Binarización adaptativa para manejar diferentes iluminaciones
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Encontrar contornos para centrar la letra
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Encontrar el contorno más grande (presumiblemente la letra)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Extraer región de interés con padding
        padding = 20
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(binary.shape[1], x + w + padding)
        y_end = min(binary.shape[0], y + h + padding)
        
        cropped = binary[y_start:y_end, x_start:x_end]
    else:
        cropped = binary
    
    # Redimensionar manteniendo aspect ratio
    h, w = cropped.shape
    if h > w:
        new_h = target_size[0]
        new_w = int(w * target_size[0] / h)
    else:
        new_w = target_size[1]
        new_h = int(h * target_size[1] / w)
    
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Crear imagen centrada con padding
    final_img = np.zeros(target_size, dtype=np.uint8)
    y_offset = (target_size[0] - new_h) // 2
    x_offset = (target_size[1] - new_w) // 2
    final_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Normalizar valores entre 0 y 1
    normalized = final_img.astype(np.float32) / 255.0
    
    return normalized

def apply_handwriting_filters(image):
    """
    Aplicar filtros específicos para análisis de escritura a mano
    """
    # Filtros direccionales para detectar trazos específicos
    # Filtro para trazos verticales (importante para letras como I, l, t)
    vertical_filter = np.array([[-1, 2, -1],
                               [-1, 2, -1],
                               [-1, 2, -1]], dtype=np.float32)
    
    # Filtro para trazos horizontales (importante para letras como H, F, E)
    horizontal_filter = np.array([[-1, -1, -1],
                                 [ 2,  2,  2],
                                 [-1, -1, -1]], dtype=np.float32)
    
    # Filtro para trazos diagonales (importante para letras como A, V, W)
    diagonal1_filter = np.array([[ 2, -1, -1],
                                [-1,  2, -1],
                                [-1, -1,  2]], dtype=np.float32)
    
    diagonal2_filter = np.array([[-1, -1,  2],
                                [-1,  2, -1],
                                [ 2, -1, -1]], dtype=np.float32)
    
    # Filtro para curvas (importante para letras como O, C, S)
    curve_filter = np.array([[ 0,  1,  0],
                            [ 1, -4,  1],
                            [ 0,  1,  0]], dtype=np.float32)
    
    # Filtro para conectores (uniones entre trazos)
    connector_filter = np.array([[ 1,  1,  1],
                                [ 1, -8,  1],
                                [ 1,  1,  1]], dtype=np.float32)
    
    # Aplicar convoluciones
    vertical_strokes = ndimage.convolve(image, vertical_filter)
    horizontal_strokes = ndimage.convolve(image, horizontal_filter)
    diagonal1_strokes = ndimage.convolve(image, diagonal1_filter)
    diagonal2_strokes = ndimage.convolve(image, diagonal2_filter)
    curves = ndimage.convolve(image, curve_filter)
    connectors = ndimage.convolve(image, connector_filter)
    
    # Combinar trazos diagonales
    diagonal_combined = np.maximum(np.abs(diagonal1_strokes), np.abs(diagonal2_strokes))
    
    return {
        'vertical_strokes': np.abs(vertical_strokes),
        'horizontal_strokes': np.abs(horizontal_strokes),
        'diagonal_strokes': diagonal_combined,
        'curves': np.abs(curves),
        'connectors': np.abs(connectors),
        'stroke_density': np.sum([np.abs(vertical_strokes), np.abs(horizontal_strokes), 
                                 diagonal_combined], axis=0)
    }

def analyze_handwriting_quality(features1, features2):
    """
    Análisis específico para calidad de escritura a mano
    """
    quality_metrics = {}
    
    # 1. Análisis de forma general
    shape_similarity = np.corrcoef(features1['stroke_density'].flatten(), 
                                  features2['stroke_density'].flatten())[0, 1]
    if np.isnan(shape_similarity):
        shape_similarity = 0.0
    quality_metrics['shape_accuracy'] = abs(shape_similarity)
    
    # 2. Análisis de trazos verticales (consistencia)
    vertical_sim = np.corrcoef(features1['vertical_strokes'].flatten(),
                              features2['vertical_strokes'].flatten())[0, 1]
    if np.isnan(vertical_sim):
        vertical_sim = 0.0
    quality_metrics['vertical_consistency'] = abs(vertical_sim)
    
    # 3. Análisis de trazos horizontales
    horizontal_sim = np.corrcoef(features1['horizontal_strokes'].flatten(),
                                features2['horizontal_strokes'].flatten())[0, 1]
    if np.isnan(horizontal_sim):
        horizontal_sim = 0.0
    quality_metrics['horizontal_consistency'] = abs(horizontal_sim)
    
    # 4. Análisis de curvas (suavidad)
    curve_sim = np.corrcoef(features1['curves'].flatten(),
                           features2['curves'].flatten())[0, 1]
    if np.isnan(curve_sim):
        curve_sim = 0.0
    quality_metrics['curve_quality'] = abs(curve_sim)
    
    # 5. Análisis de proporciones
    density1 = np.sum(features1['stroke_density'] > 0.1)
    density2 = np.sum(features2['stroke_density'] > 0.1)
    proportion_diff = abs(density1 - density2) / max(density1, density2, 1)
    quality_metrics['proportion_accuracy'] = max(0, 1 - proportion_diff)
    
    return quality_metrics

def generate_feedback(precision, quality_metrics):
    """
    Generar retroalimentación educativa para el estudiante
    """
    feedback = {
        'overall_score': precision,
        'level': '',
        'strengths': [],
        'areas_for_improvement': [],
        'specific_tips': []
    }
    
    # Determinar nivel
    if precision >= 0.9:
        feedback['level'] = 'Excelente'
        feedback['strengths'].append('Escritura muy precisa y bien formada')
    elif precision >= 0.75:
        feedback['level'] = 'Muy Bueno'
        feedback['strengths'].append('Buena formación de letra')
    elif precision >= 0.6:
        feedback['level'] = 'Bueno'
        feedback['strengths'].append('Forma reconocible de la letra')
    elif precision >= 0.4:
        feedback['level'] = 'En Desarrollo'
        feedback['areas_for_improvement'].append('Necesita mejorar la forma general')
    else:
        feedback['level'] = 'Necesita Práctica'
        feedback['areas_for_improvement'].append('Requiere práctica adicional')
    
    # Análisis específico por componentes
    if quality_metrics['vertical_consistency'] < 0.6:
        feedback['areas_for_improvement'].append('Mejorar trazos verticales')
        feedback['specific_tips'].append('Practica líneas rectas de arriba hacia abajo')
    
    if quality_metrics['horizontal_consistency'] < 0.6:
        feedback['areas_for_improvement'].append('Mejorar trazos horizontales')
        feedback['specific_tips'].append('Practica líneas horizontales de izquierda a derecha')
    
    if quality_metrics['curve_quality'] < 0.6:
        feedback['areas_for_improvement'].append('Mejorar curvas y círculos')
        feedback['specific_tips'].append('Practica movimientos circulares suaves')
    
    if quality_metrics['proportion_accuracy'] < 0.7:
        feedback['areas_for_improvement'].append('Ajustar proporciones de la letra')
        feedback['specific_tips'].append('Observa el tamaño y espaciado de la letra modelo')
    
    # Agregar fortalezas específicas
    if quality_metrics['vertical_consistency'] >= 0.8:
        feedback['strengths'].append('Excelentes trazos verticales')
    if quality_metrics['horizontal_consistency'] >= 0.8:
        feedback['strengths'].append('Buenos trazos horizontales')
    if quality_metrics['curve_quality'] >= 0.8:
        feedback['strengths'].append('Curvas bien ejecutadas')
    
    return feedback

def calculate_precision(original_path, compare_path):
    """
    Calcular precisión entre dos imágenes de escritura a mano
    """
    try:
        # Preprocesar imágenes optimizado para escritura
        img1 = preprocess_image(original_path)
        img2 = preprocess_image(compare_path)
        
        # Aplicar filtros específicos para escritura a mano
        features1 = apply_handwriting_filters(img1)
        features2 = apply_handwriting_filters(img2)
        
        # Análizar calidad de escritura
        quality_metrics = analyze_handwriting_quality(features1, features2)
        
        # Calcular SSIM con parámetros optimizados para escritura
        ssim_score = ssim(img1, img2, data_range=1.0, win_size=7)
        
        # Calcular similitud estructural específica para letras
        mse = np.mean((img1 - img2) ** 2)
        structural_similarity = 1 / (1 + mse * 10)  # Penalizar más diferencias
        
        # Pesos ajustados para evaluación de escritura
        weights = {
            'shape_accuracy': 0.25,        # Forma general más importante
            'vertical_consistency': 0.20,  # Trazos verticales críticos
            'horizontal_consistency': 0.15, # Trazos horizontales
            'curve_quality': 0.15,         # Calidad de curvas
            'proportion_accuracy': 0.15,   # Proporciones correctas
            'ssim': 0.10                   # Similitud estructural general
        }
        
        # Calcular puntuación final ponderada
        total_score = 0
        for metric, weight in weights.items():
            if metric in quality_metrics:
                total_score += quality_metrics[metric] * weight
            elif metric == 'ssim':
                total_score += abs(ssim_score) * weight
        
        # Asegurar que el resultado esté entre 0 y 1
        precision = max(0.0, min(1.0, total_score))
        
        # Generar retroalimentación educativa
        feedback = generate_feedback(precision, quality_metrics)
        
        return {
            'precision': round(precision, 4),
            'feedback': feedback,
            'technical_analysis': {
                'quality_metrics': {k: round(v, 4) for k, v in quality_metrics.items()},
                'ssim_score': round(abs(ssim_score), 4),
                'structural_similarity': round(structural_similarity, 4)
            }
        }
        
    except Exception as e:
        raise Exception(f"Error en el análisis de escritura: {str(e)}")

@app.route('/image/compare', methods=['POST'])
def compare_images():
    """
    Endpoint principal para comparar imágenes
    """
    try:
        # Verificar que se enviaron archivos
        if 'original_img' not in request.files or 'compare_img' not in request.files:
            return jsonify({
                'error': 'Se requieren ambos archivos: original_img y compare_img'
            }), 400
        
        original_file = request.files['original_img']
        compare_file = request.files['compare_img']
        
        # Verificar que los archivos no estén vacíos
        if original_file.filename == '' or compare_file.filename == '':
            return jsonify({
                'error': 'Ambos archivos deben tener nombres válidos'
            }), 400
        
        # Verificar extensiones de archivo
        if not (allowed_file(original_file.filename) and allowed_file(compare_file.filename)):
            return jsonify({
                'error': 'Formato de archivo no permitido. Use: png, jpg, jpeg, gif, bmp'
            }), 400
        
        # Guardar archivos temporalmente
        original_filename = secure_filename(original_file.filename)
        compare_filename = secure_filename(compare_file.filename)
        
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_original_{original_filename}")
        compare_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_compare_{compare_filename}")
        
        original_file.save(original_path)
        compare_file.save(compare_path)
        
        try:
            # Calcular precisión
            result = calculate_precision(original_path, compare_path)
            
            response = {
                'success': True,
                'precision': result['precision'],
                'score_percentage': f"{result['precision']*100:.1f}%",
                'educational_feedback': result['feedback'],
                'algorithm_details': {
                    'method': 'Análisis de escritura con filtros de convolución especializados',
                    'techniques_used': [
                        'Detección de trazos verticales y horizontales',
                        'Análisis de curvas y conectores',
                        'Evaluación de proporciones',
                        'Binarización adaptativa',
                        'Centrado automático de letras'
                    ],
                    'optimized_for': 'Evaluación de escritura a mano en alfabetización'
                },
                'technical_analysis': result['technical_analysis']
            }
            
            return jsonify(response), 200
            
        finally:
            # Limpiar archivos temporales
            try:
                os.remove(original_path)
                os.remove(compare_path)
            except:
                pass
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error interno del servidor: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado del servicio"""
    return jsonify({
        'status': 'healthy',
        'service': 'Image Comparison Service',
        'version': '1.0.0'
    }), 200

@app.route('/', methods=['GET'])
def info():
    """Información del servicio"""
    return jsonify({
        'service': 'Servicio de Evaluación de Escritura para Alfabetización',
        'description': 'API especializada para evaluar escritura a mano usando algoritmos de convolución',
        'target_audience': 'Jóvenes y adultos en proceso de alfabetización',
        'endpoints': {
            'POST /image/compare': 'Comparar escritura del estudiante con letra modelo',
            'GET /health': 'Verificar estado del servicio',
            'GET /': 'Información del servicio'
        },
        'parameters': {
            'original_img': 'Imagen de la letra modelo/correcta (File)',
            'compare_img': 'Imagen de la escritura del estudiante (File)'
        },
        'educational_features': [
            'Retroalimentación específica por tipo de trazo',
            'Identificación de fortalezas del estudiante',
            'Sugerencias personalizadas de mejora',
            'Análisis de consistencia en trazos',
            'Evaluación de proporciones'
        ],
        'supported_formats': ['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        'max_file_size': '16MB',
        'recommendations': [
            'Use imágenes con fondo blanco y tinta negra para mejores resultados',
            'Centre la letra en la imagen',
            'Evite sombras o reflejos en la fotografía',
            'Asegúrese de que la letra esté completamente visible'
        ]
    }), 200

if __name__ == '__main__':
    print("Iniciando Servicio de Evaluación de Escritura para Alfabetización")
    print("Algoritmos optimizados para análisis de escritura a mano")
    print("Diseñado para jóvenes y adultos en proceso de alfabetización")
    print("Servidor disponible en: http://localhost:5001")
    print("Documentación en: http://localhost:5001")
    print("Características especiales:")
    print("   - Análisis de trazos direccionales")
    print("   - Retroalimentación educativa personalizada")
    print("   - Detección automática de letras")
    print("   - Evaluación de proporciones y consistencia")
    
    app.run(debug=True, host='0.0.0.0', port=5001)