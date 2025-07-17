from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
from letter_comparator import LetterComparator
from models import AnalysisResults

app = Flask(__name__)

# Configuración
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp'}

# Instancia global del comparador
comparator = LetterComparator()

def allowed_file(filename):
    """Verifica si el archivo tiene una extensión permitida"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_level_description(score):
    """Convierte el score numérico a descripción textual"""
    if score >= 0.8:
        return {"level": "Excelente"}
    elif score >= 0.6:
        return {"level": "Bueno"}
    elif score >= 0.4:
        return {"level": "Necesita mejora"}
    else:
        return {"level": "Requiere práctica"}

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de verificación de salud"""
    return jsonify({
        "status": "healthy",
        "message": "Letter Analysis API is running"
    }), 200

@app.route('/image/compare', methods=['POST'])
def compare_images():
    """
    Endpoint principal para comparar dos imágenes de letras
    
    Espera:
    - original_img: archivo de imagen (la imagen del usuario)
    - compare_img: archivo de imagen (la plantilla de referencia)
    
    Retorna:
    - JSON con análisis completo y recomendaciones
    """
    try:
        # Verificar que se enviaron ambas imágenes
        if 'original_img' not in request.files or 'compare_img' not in request.files:
            return jsonify({
                "error": "Se requieren ambas imágenes: 'original_img' y 'compare_img'"
            }), 400
        
        original_file = request.files['original_img']
        compare_file = request.files['compare_img']
        
        # Verificar que los archivos no estén vacíos
        if original_file.filename == '' or compare_file.filename == '':
            return jsonify({
                "error": "Los archivos no pueden estar vacíos"
            }), 400
        
        # Verificar extensiones permitidas
        if not (allowed_file(original_file.filename) and allowed_file(compare_file.filename)):
            return jsonify({
                "error": f"Extensiones permitidas: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Leer los bytes de las imágenes directamente
        original_bytes = original_file.read()
        compare_bytes = compare_file.read()
        
        # Verificar que los archivos no estén vacíos
        if len(original_bytes) == 0 or len(compare_bytes) == 0:
            return jsonify({
                "error": "Los archivos están vacíos"
            }), 400
        
        # Realizar el análisis usando bytes directamente
        results = comparator.compare_letters_from_bytes(original_bytes, compare_bytes)
        
        # Preparar respuesta estructurada
        response = {
            "success": True,
            "analysis": {
                "scores": {
                    "ssim_score": {
                        "value": round(results.ssim_score, 3),
                        "name": "Similitud General (SSIM)",
                        "description": "Qué tan similar es la forma general a la plantilla",
                        **get_level_description(results.ssim_score)
                    },
                    "stroke_quality": {
                        "value": round(results.stroke_quality, 3),
                        "name": "Calidad del Trazo",
                        "description": "Continuidad y completitud del trazo",
                        **get_level_description(results.stroke_quality)
                    },
                    "curve_smoothness": {
                        "value": round(results.curve_smoothness, 3),
                        "name": "Suavidad de Curvas",
                        "description": "Qué tan fluidas son las curvas",
                        **get_level_description(results.curve_smoothness)
                    },
                    "line_straightness": {
                        "value": round(results.line_straightness, 3),
                        "name": "Rectitud de Líneas",
                        "description": "Precisión en líneas rectas",
                        **get_level_description(results.line_straightness)
                    },
                    "thickness_uniformity": {
                        "value": round(results.thickness_uniformity, 3),
                        "name": "Uniformidad del Grosor",
                        "description": "Consistencia en el grosor del trazo",
                        **get_level_description(results.thickness_uniformity)
                    },
                    "corner_quality": {
                        "value": round(results.corner_quality, 3),
                        "name": "Calidad de Esquinas",
                        "description": "Definición de esquinas y puntos críticos",
                        **get_level_description(results.corner_quality)
                    }
                },
                "overall_score": {
                    "value": round(results.overall_score, 3),
                    "name": "Puntuación General",
                    "description": "Score general ponderado",
                    **get_level_description(results.overall_score)
                },
                "recommendations": results.recommendations,
                "summary": {
                    "total_aspects": 6,
                    "excellent_count": sum(1 for score in [
                        results.ssim_score, results.stroke_quality, results.curve_smoothness,
                        results.line_straightness, results.thickness_uniformity, results.corner_quality
                    ] if score >= 0.8),
                    "good_count": sum(1 for score in [
                        results.ssim_score, results.stroke_quality, results.curve_smoothness,
                        results.line_straightness, results.thickness_uniformity, results.corner_quality
                    ] if 0.6 <= score < 0.8),
                    "needs_improvement_count": sum(1 for score in [
                        results.ssim_score, results.stroke_quality, results.curve_smoothness,
                        results.line_straightness, results.thickness_uniformity, results.corner_quality
                    ] if 0.4 <= score < 0.6),
                    "needs_practice_count": sum(1 for score in [
                        results.ssim_score, results.stroke_quality, results.curve_smoothness,
                        results.line_straightness, results.thickness_uniformity, results.corner_quality
                    ] if score < 0.4)
                }
            },
            "metadata": {
                "original_filename": secure_filename(original_file.filename),
                "compare_filename": secure_filename(compare_file.filename),
                "original_size_bytes": len(original_bytes),
                "compare_size_bytes": len(compare_bytes)
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error en compare_images: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Error interno del servidor: {str(e)}"
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Manejo de archivos demasiado grandes"""
    return jsonify({
        "error": "El archivo es demasiado grande. Máximo 16MB permitido."
    }), 413

@app.errorhandler(400)
def bad_request(e):
    """Manejo de solicitudes malformadas"""
    return jsonify({
        "error": "Solicitud malformada"
    }), 400

@app.errorhandler(500)
def internal_error(e):
    """Manejo de errores internos"""
    return jsonify({
        "error": "Error interno del servidor"
    }), 500

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )