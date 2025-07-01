from flask import Flask, request, jsonify
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageEnhance, ImageFilter

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB máximo

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
contador = 0  # contador global

# Crear directorio temporal si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Verificar si el archivo tiene una extensión permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_original_image(image_path, target_size=(128, 128)):
    """
    Imagen original SIN procesamiento - queda tal como viene
    Solo se lee, redimensiona y normaliza
    """
    global contador 
    
    # Leer imagen
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Redimensionar directamente al tamaño objetivo
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalizar valores entre 0 y 1
    normalized = resized.astype(np.float32) / 255.0
    
    # Guardar imagen procesada para debug
    nombre = f"original_{contador}"
    cv2.imwrite(f"/Users/rextro/Documents/ICHEJA/AC_Comparar_Imagen/imagen/{nombre}.png", resized)
    
    return normalized

def preprocess_compare_image(image_path, target_size=(128, 128)):
    """
    Procesar imagen de comparación ÚNICAMENTE con filtros específicos:
    - Escala de grises
    - Contraste: 50%
    - Brillo: 50%
    - Detalle (sharpen): 100%
    - Recortar y centrar al mismo tamaño que la original
    """
    global contador 
    
    try:
        # Leer imagen usando PIL para mejor control de filtros
        pil_img = Image.open(image_path)
        
        # 1. Convertir a escala de grises
        if pil_img.mode != 'L':
            pil_img = pil_img.convert('L')
        
        # 2. Ajustar brillo al 50%
        brightness_enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = brightness_enhancer.enhance(1.5)  # 50% más brillo
        
        # 3. Ajustar contraste al 50%
        contrast_enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = contrast_enhancer.enhance(1.5)  # 50% más contraste
        
        # 4. Aplicar filtro de detalle (sharpen) al 100%
        pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))
        
        # 5. Convertir a array numpy
        img_array = np.array(pil_img)
        
        # 6. Redimensionar al tamaño objetivo (mismo que la original)
        resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_AREA)
        
        # 7. Normalizar valores entre 0 y 1
        normalized = resized.astype(np.float32) / 255.0
        
        # Guardar imagen procesada para debug
        nombre = f"comparar_{contador}"
        cv2.imwrite(f"/Users/rextro/Documents/ICHEJA/AC_Comparar_Imagen/imagen/{nombre}.png", resized)
        contador += 1
        
        return normalized
        
    except Exception as e:
        raise ValueError(f"Error procesando imagen de comparación: {str(e)}")

# =====================================
# FUNCIONES DE SIMILITUD
# =====================================

def compare_contours(img1, img2):
    """Comparar similitud basada en contornos"""
    try:
        # Convertir a binario para encontrar contornos
        _, binary1 = cv2.threshold((img1 * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        _, binary2 = cv2.threshold((img2 * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        
        # Encontrar contornos
        contours1, _ = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours1 or not contours2:
            return 0.0
        
        # Obtener el contorno más grande de cada imagen
        largest_contour1 = max(contours1, key=cv2.contourArea)
        largest_contour2 = max(contours2, key=cv2.contourArea)
        
        # Comparar usando matchShapes (método Hu moments)
        similarity = cv2.matchShapes(largest_contour1, largest_contour2, cv2.CONTOURS_MATCH_I1, 0)
        
        # Convertir a similitud (0=idéntico, mayor=más diferente)
        return max(0.0, 1.0 - min(similarity, 1.0))
        
    except:
        return 0.0

def compare_histograms(img1, img2):
    """Comparar histogramas de intensidad"""
    try:
        # Calcular histogramas
        hist1 = cv2.calcHist([(img1 * 255).astype(np.uint8)], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([(img2 * 255).astype(np.uint8)], [0], None, [256], [0, 256])
        
        # Normalizar histogramas
        hist1 = hist1.flatten() / np.sum(hist1)
        hist2 = hist2.flatten() / np.sum(hist2)
        
        # Usar correlación para comparar
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return max(0.0, correlation)
        
    except:
        return 0.0

def template_matching_similarity(img1, img2):
    """Template matching directo"""
    try:
        # Convertir a uint8
        template = (img1 * 255).astype(np.uint8)
        image = (img2 * 255).astype(np.uint8)
        
        # Template matching
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        
        # Obtener el valor máximo
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        return max(0.0, max_val)
        
    except:
        return 0.0

def compare_pixel_density(img1, img2):
    """Comparar densidad de píxeles activos"""
    try:
        # Calcular píxeles activos (>threshold)
        threshold = 0.1
        active1 = np.sum(img1 > threshold)
        active2 = np.sum(img2 > threshold)
        
        # Calcular similitud de densidad
        if max(active1, active2) == 0:
            return 1.0 if active1 == active2 else 0.0
        
        density_ratio = min(active1, active2) / max(active1, active2)
        
        return density_ratio
        
    except:
        return 0.0

def calculate_image_similarity(img1, img2):
    """
    Calcula múltiples métricas de similitud entre dos imágenes
    Enfocado en similitud visual, no en calidad de trazos
    """
    
    similarity_metrics = {}
    
    # 1. SSIM (Structural Similarity Index) - LA MÁS EFECTIVA
    ssim_score = ssim(img1, img2, data_range=1.0, win_size=7)
    similarity_metrics['ssim'] = abs(ssim_score)
    
    # 2. Correlación cruzada normalizada
    correlation = np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    similarity_metrics['correlation'] = abs(correlation)
    
    # 3. Template matching directo
    template_similarity = template_matching_similarity(img1, img2)
    similarity_metrics['template_matching'] = template_similarity
    
    # 4. Comparación de contornos
    contour_similarity = compare_contours(img1, img2)
    similarity_metrics['contour_similarity'] = contour_similarity
    
    # 5. Similitud de histogramas
    hist_similarity = compare_histograms(img1, img2)
    similarity_metrics['histogram_similarity'] = hist_similarity
    
    # 6. Mean Squared Error (convertido a similitud)
    mse = np.mean((img1 - img2) ** 2)
    mse_similarity = 1 / (1 + mse * 10)  # Convertir error a similitud
    similarity_metrics['mse_similarity'] = mse_similarity
    
    # 7. Comparación de densidad de píxeles
    density_similarity = compare_pixel_density(img1, img2)
    similarity_metrics['pixel_density'] = density_similarity
    
    return similarity_metrics

def generate_similarity_feedback(similarity_score, detailed_metrics):
    """
    Generar retroalimentación basada en similitud visual
    """
    feedback = {
        'overall_score': similarity_score,
        'similarity_level': '',
        'visual_analysis': [],
        'recommendations': [],
        'confidence': ''
    }
    
    # Determinar nivel de similitud
    if similarity_score >= 0.9:
        feedback['similarity_level'] = 'Muy Similar'
        feedback['confidence'] = 'Alta'
        feedback['visual_analysis'].append('Las imágenes son prácticamente idénticas')
    elif similarity_score >= 0.8:
        feedback['similarity_level'] = 'Bastante Similar'
        feedback['confidence'] = 'Alta'
        feedback['visual_analysis'].append('Las imágenes tienen gran parecido visual')
    elif similarity_score >= 0.6:
        feedback['similarity_level'] = 'Similar'
        feedback['confidence'] = 'Media'
        feedback['visual_analysis'].append('Las imágenes comparten características principales')
    elif similarity_score >= 0.4:
        feedback['similarity_level'] = 'Poco Similar'
        feedback['confidence'] = 'Media'
        feedback['visual_analysis'].append('Las imágenes tienen algunas diferencias notables')
    else:
        feedback['similarity_level'] = 'Muy Diferente'
        feedback['confidence'] = 'Baja'
        feedback['visual_analysis'].append('Las imágenes son significativamente diferentes')
    
    # Análisis específico por métricas
    if detailed_metrics.get('ssim', {}).get('value', 0) > 0.8:
        feedback['visual_analysis'].append('Excelente similitud estructural')
    elif detailed_metrics.get('ssim', {}).get('value', 0) < 0.5:
        feedback['recommendations'].append('La estructura general de las imágenes difiere considerablemente')
    
    if detailed_metrics.get('correlation', {}).get('value', 0) > 0.8:
        feedback['visual_analysis'].append('Muy buena correlación pixel a pixel')
    
    if detailed_metrics.get('contour_similarity', {}).get('value', 0) > 0.7:
        feedback['visual_analysis'].append('Las formas son muy parecidas')
    elif detailed_metrics.get('contour_similarity', {}).get('value', 0) < 0.4:
        feedback['recommendations'].append('Las formas generales difieren significativamente')
    
    if detailed_metrics.get('template_matching', {}).get('value', 0) > 0.8:
        feedback['visual_analysis'].append('Excelente coincidencia de patrones')
    
    return feedback

def calculate_precision(original_path, compare_path):
    """
    Calcular similitud entre dos imágenes usando múltiples estrategias
    """
    try:
        # Preprocesar imágenes
        img1 = preprocess_original_image(original_path)      # Imagen original (sin filtros)
        img2 = preprocess_compare_image(compare_path)        # Imagen a comparar (filtros específicos)
        
        # Calcular todas las métricas de similitud
        metrics = calculate_image_similarity(img1, img2)
        
        # PESOS OPTIMIZADOS PARA SIMILITUD VISUAL
        weights = {
            'ssim': 0.15,                    # Similitud estructural (muy importante)
            'correlation': 0.25,             # Correlación directa
            'template_matching': 0.20,       # Template matching
            'contour_similarity': 0.30,      # Similitud de forma
            'histogram_similarity': 0.05,    # Distribución de intensidades
            'mse_similarity': 0.03,          # Error cuadrático medio
            'pixel_density': 0.02            # Densidad de píxeles
        }
        
        # Calcular puntuación final ponderada
        total_score = 0.0
        detailed_scores = {}
        
        print("=== CÁLCULO DE SIMILITUD ===")
        for metric, weight in weights.items():
            if metric in metrics:
                score_contribution = metrics[metric] * weight
                print(f"Calculando {metric} con peso {weight} = {metrics[metric]:.4f} * {weight:.4f} = {score_contribution:.4f}")
                total_score += score_contribution
                detailed_scores[metric] = {
                    'value': round(metrics[metric], 4),
                    'weight': weight,
                    'contribution': round(score_contribution, 4)
                }
                print(f"{metric}: {metrics[metric]:.4f} * {weight} = {score_contribution:.4f}")
        
        print(f"Total Score: {total_score:.4f}")
        
        # Asegurar que el resultado esté entre 0 y 1
        similarity = max(0.0, min(1.0, total_score))
        
        # Generar retroalimentación basada en similitud
        feedback = generate_similarity_feedback(similarity, detailed_scores)
        
        return {
            'precision': round(similarity, 4),
            'feedback': feedback,
            'technical_analysis': {
                'detailed_metrics': detailed_scores,
                'raw_metrics': {k: round(v, 4) for k, v in metrics.items()},
                'total_weighted_score': round(total_score, 4)
            }
        }
        
    except Exception as e:
        raise Exception(f"Error en el análisis de similitud: {str(e)}")

@app.route('/image/compare', methods=['POST'])
def compare_images():
    """
    Endpoint principal para comparar similitud de imágenes
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
            # Calcular similitud
            result = calculate_precision(original_path, compare_path)
            
            response = {
                'success': True,
                'similarity': result['precision'],
                'similarity_percentage': f"{result['precision']*100:.1f}%",
                'visual_feedback': result['feedback'],
                'algorithm_details': {
                    'method': 'Análisis de similitud visual multi-métrica',
                    'focus': 'Similitud de imágenes (no calidad de trazos)',
                    'techniques_used': [
                        'SSIM (Structural Similarity Index)',
                        'Correlación cruzada normalizada',
                        'Template matching directo',
                        'Comparación de contornos',
                        'Análisis de histogramas',
                        'Comparación de densidad de píxeles'
                    ],
                    'original_processing': [
                        'Sin filtros - imagen tal como viene',
                        'Solo redimensionado y normalización'
                    ],
                    'compare_processing': [
                        'Escala de grises',
                        'Brillo: +50%',
                        'Contraste: +50%',
                        'Detalle (sharpen): 100%',
                        'Redimensionado al mismo tamaño'
                    ],
                    'optimized_for': 'Comparación visual directa de similitud'
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
        'service': 'Image Similarity Comparison Service',
        'version': '2.0.0 - Visual Similarity Focus'
    }), 200

@app.route('/', methods=['GET'])
def info():
    """Información del servicio"""
    return jsonify({
        'service': 'Servicio de Comparación de Similitud Visual',
        'description': 'API especializada para medir similitud visual entre imágenes',
        'version': '2.0.0 - Visual Similarity Focus',
        'focus': 'Similitud visual (no calidad de trazos)',
        'target_audience': 'Comparación directa de similitud entre imágenes',
        'endpoints': {
            'POST /image/compare': 'Comparar similitud visual entre dos imágenes',
            'GET /health': 'Verificar estado del servicio',
            'GET /': 'Información del servicio'
        },
        'parameters': {
            'original_img': 'Imagen de referencia (File)',
            'compare_img': 'Imagen a comparar (File)'
        },
        'similarity_metrics': [
            'SSIM (Structural Similarity Index) - 30%',
            'Correlación cruzada - 25%',
            'Template matching - 20%',
            'Similitud de contornos - 15%',
            'Análisis de histogramas - 5%',
            'Error cuadrático medio - 3%',
            'Densidad de píxeles - 2%'
        ],
        'processing_methods': {
            'original_image': 'Sin filtros - imagen queda tal como viene',
            'compare_image': 'Filtros específicos: escala grises, brillo +50%, contraste +50%, sharpen 100%'
        },
        'output_format': {
            'similarity': 'Puntuación de 0.0 a 1.0',
            'similarity_percentage': 'Porcentaje de similitud',
            'visual_feedback': 'Análisis descriptivo de la similitud',
            'technical_analysis': 'Métricas detalladas por técnica'
        },
        'supported_formats': ['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        'max_file_size': '16MB',
        'recommendations': [
            'Use imágenes claras y bien definidas',
            'Mantenga las imágenes centradas',
            'Evite ruido visual excesivo',
            'Las imágenes se redimensionarán automáticamente al mismo tamaño'
        ]
    }), 200

if __name__ == '__main__':
    print("Iniciando Servicio de Comparación de Similitud Visual")
    print("VERSIÓN 2.0.0 - Enfoque en Similitud Visual")
    print("Algoritmos optimizados para medir similitud entre imágenes")
    print("Diseñado para comparación directa de similitud visual")
    print("Servidor disponible en: http://localhost:5001")
    print("Documentación en: http://localhost:5001")
    print("\nCaracterísticas principales:")
    print("   - Enfoque en SIMILITUD VISUAL (no calidad de trazos)")
    print("   - SSIM (Structural Similarity Index) - Métrica principal")
    print("   - Correlación cruzada normalizada")
    print("   - Template matching directo")
    print("   - Comparación de contornos y formas")
    print("   - Análisis de histogramas")
    print("   - Múltiples métricas combinadas con pesos optimizados")
    print("\nProcesamiento:")
    print("   - Imagen original: SIN filtros (tal como viene)")
    print("   - Imagen comparación: Filtros específicos para optimizar similitud")
    
    app.run(debug=True, host='0.0.0.0', port=5001)