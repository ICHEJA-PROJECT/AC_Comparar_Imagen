import numpy as np
import math
from typing import List
from skimage.metrics import structural_similarity as ssim

from models import AnalysisResults
from image_preprocessor import ImagePreprocessor
from stroke_analyzer import StrokeAnalyzer

class LetterComparator:
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.stroke_analyzer = StrokeAnalyzer()
    
    def compare_letters(self, user_image_path: str, template_image_path: str) -> AnalysisResults:
        try:
            # Preprocesar imágenes
            user_img = self.preprocessor.preprocess(user_image_path)
            template_img = self.preprocessor.preprocess(template_image_path)
            
            # Validar que las imágenes no estén vacías
            if np.sum(user_img) == 0:
                print("Advertencia: La imagen del usuario está vacía después del preprocesamiento")
            if np.sum(template_img) == 0:
                print("Advertencia: La imagen de la plantilla está vacía después del preprocesamiento")
            
            # Análisis SSIM básico
            ssim_score, _ = ssim(user_img, template_img, full=True)
            
            # Análisis detallado del trazo del usuario
            stroke_quality = self.stroke_analyzer.analyze_stroke_quality(user_img)
            curve_smoothness = self.stroke_analyzer.analyze_curve_smoothness(user_img)
            line_straightness = self.stroke_analyzer.analyze_line_straightness(user_img)
            thickness_uniformity = self.stroke_analyzer.analyze_thickness_uniformity(user_img)
            corner_quality = self.stroke_analyzer.analyze_corner_quality(user_img)
            
            # Debug: imprimir valores antes de procesar
            print(f"Debug - Valores raw:")
            print(f"  SSIM: {ssim_score:.3f}")
            print(f"  Stroke: {stroke_quality:.3f}")
            print(f"  Curves: {curve_smoothness:.3f}")
            print(f"  Lines: {line_straightness:.3f}")
            print(f"  Thickness: {thickness_uniformity:.3f}")
            print(f"  Corners: {corner_quality:.3f}")
            
            # Aplicar función sigmoide a cada métrica individualmente
            (ssim_final, stroke_final, curves_final, 
             lines_final, thickness_final, corners_final) = self._apply_sigmoid_to_all_metrics(
                ssim_score, stroke_quality, curve_smoothness,
                line_straightness, thickness_uniformity, corner_quality
            )
            
            # Debug: imprimir valores después de sigmoide
            print(f"Debug - Valores después de sigmoide:")
            print(f"  SSIM: {ssim_final:.3f}")
            print(f"  Stroke: {stroke_final:.3f}")
            print(f"  Curves: {curves_final:.3f}")
            print(f"  Lines: {lines_final:.3f}")
            print(f"  Thickness: {thickness_final:.3f}")
            print(f"  Corners: {corners_final:.3f}")
            
            # Score general ponderado (ya no necesita sigmoide adicional)
            overall_score = self._calculate_overall_score(
                ssim_score, stroke_quality, curve_smoothness, 
                line_straightness, thickness_uniformity, corner_quality
            )
            
            # Generar recomendaciones basadas en los valores sigmoidizados
            recommendations = self._generate_recommendations(
                ssim_final, stroke_final, curves_final,
                lines_final, thickness_final, corners_final
            )
            
            return AnalysisResults(
                ssim_score=float(np.clip(ssim_final, 0, 1)),
                stroke_quality=float(np.clip(stroke_final, 0, 1)),
                curve_smoothness=float(np.clip(curves_final, 0, 1)),
                line_straightness=float(np.clip(lines_final, 0, 1)),
                thickness_uniformity=float(np.clip(thickness_final, 0, 1)),
                corner_quality=float(np.clip(corners_final, 0, 1)),
                overall_score=float(np.clip(overall_score, 0, 1)),
                recommendations=recommendations
            )
            
        except Exception as e:
            print(f"Error en el análisis: {e}")
            # Retornar valores por defecto en caso de error
            return AnalysisResults(
                ssim_score=0.0,
                stroke_quality=0.0,
                curve_smoothness=0.0,
                line_straightness=0.0,
                thickness_uniformity=0.0,
                corner_quality=0.0,
                overall_score=0.0,
                recommendations=["Error en el análisis. Verificar las imágenes de entrada."]
            )
    
    def compare_letters_from_bytes(self, user_image_bytes: bytes, template_image_bytes: bytes) -> AnalysisResults:
        try:
            # Preprocesar imágenes desde bytes
            user_img = self.preprocessor.preprocess_from_bytes(user_image_bytes)
            template_img = self.preprocessor.preprocess_from_bytes(template_image_bytes)
            
            # Validar que las imágenes no estén vacías
            if np.sum(user_img) == 0:
                print("Advertencia: La imagen del usuario está vacía después del preprocesamiento")
            if np.sum(template_img) == 0:
                print("Advertencia: La imagen de la plantilla está vacía después del preprocesamiento")
            
            # Análisis SSIM básico
            ssim_score, _ = ssim(user_img, template_img, full=True)
            
            # Análisis detallado del trazo del usuario
            stroke_quality = self.stroke_analyzer.analyze_stroke_quality(user_img)
            curve_smoothness = self.stroke_analyzer.analyze_curve_smoothness(user_img)
            line_straightness = self.stroke_analyzer.analyze_line_straightness(user_img)
            thickness_uniformity = self.stroke_analyzer.analyze_thickness_uniformity(user_img)
            corner_quality = self.stroke_analyzer.analyze_corner_quality(user_img)
            
            # Debug: imprimir valores antes de procesar
            print(f"Debug - Valores raw:")
            print(f"  SSIM: {ssim_score:.3f}")
            print(f"  Stroke: {stroke_quality:.3f}")
            print(f"  Curves: {curve_smoothness:.3f}")
            print(f"  Lines: {line_straightness:.3f}")
            print(f"  Thickness: {thickness_uniformity:.3f}")
            print(f"  Corners: {corner_quality:.3f}")
            
            # Aplicar función sigmoide a cada métrica individualmente
            (ssim_final, stroke_final, curves_final, 
             lines_final, thickness_final, corners_final) = self._apply_sigmoid_to_all_metrics(
                ssim_score, stroke_quality, curve_smoothness,
                line_straightness, thickness_uniformity, corner_quality
            )
            
            # Debug: imprimir valores después de sigmoide
            print(f"Debug - Valores después de sigmoide:")
            print(f"  SSIM: {ssim_final:.3f}")
            print(f"  Stroke: {stroke_final:.3f}")
            print(f"  Curves: {curves_final:.3f}")
            print(f"  Lines: {lines_final:.3f}")
            print(f"  Thickness: {thickness_final:.3f}")
            print(f"  Corners: {corners_final:.3f}")
            
            # Score general ponderado (ya no necesita sigmoide adicional)
            overall_score = self._calculate_overall_score(
                ssim_score, stroke_quality, curve_smoothness, 
                line_straightness, thickness_uniformity, corner_quality
            )
            
            # Generar recomendaciones basadas en los valores sigmoidizados
            recommendations = self._generate_recommendations(
                ssim_final, stroke_final, curves_final,
                lines_final, thickness_final, corners_final
            )
            
            return AnalysisResults(
                ssim_score=float(np.clip(ssim_final, 0, 1)),
                stroke_quality=float(np.clip(stroke_final, 0, 1)),
                curve_smoothness=float(np.clip(curves_final, 0, 1)),
                line_straightness=float(np.clip(lines_final, 0, 1)),
                thickness_uniformity=float(np.clip(thickness_final, 0, 1)),
                corner_quality=float(np.clip(corners_final, 0, 1)),
                overall_score=float(np.clip(overall_score, 0, 1)),
                recommendations=recommendations
            )
            
        except Exception as e:
            print(f"Error en el análisis: {e}")
            # Retornar valores por defecto en caso de error
            return AnalysisResults(
                ssim_score=0.0,
                stroke_quality=0.0,
                curve_smoothness=0.0,
                line_straightness=0.0,
                thickness_uniformity=0.0,
                corner_quality=0.0,
                overall_score=0.0,
                recommendations=["Error en el análisis. Verificar las imágenes de entrada."]
            )
    
    def _calculate_overall_score(self, ssim_score: float, stroke_quality: float,
                                curve_smoothness: float, line_straightness: float,
                                thickness_uniformity: float, corner_quality: float) -> float:
        """Calcula score general ponderado aplicando sigmoide a cada parámetro"""
        
        # Aplicar función sigmoide a cada parámetro individualmente
        # con diferentes centros y pendientes según la importancia de cada métrica
        ssim_sigmoid = self._sigmoid_score(ssim_score, center=0.4, k=10)
        stroke_sigmoid = self._sigmoid_score(stroke_quality, center=0.3, k=12)
        curves_sigmoid = self._sigmoid_score(curve_smoothness, center=0.4, k=10)
        lines_sigmoid = self._sigmoid_score(line_straightness, center=0.3, k=12)
        thickness_sigmoid = self._sigmoid_score(thickness_uniformity, center=0.5, k=8)
        corners_sigmoid = self._sigmoid_score(corner_quality, center=0.5, k=8)
        
        # Crear weights con valores sigmoidizados ponderados
        weights = {
            'ssim_sigmoid': ssim_sigmoid * 0.5,        # Aumentado de 0.3 a 0.5
            'stroke_sigmoid': stroke_sigmoid * 0.15,   # Reducido de 0.2 a 0.15
            'curves_sigmoid': curves_sigmoid * 0.15,   # Reducido de 0.2 a 0.15
            'lines_sigmoid': lines_sigmoid * 0.1,      # Reducido de 0.15 a 0.1
            'thickness_sigmoid': thickness_sigmoid * 0.05,  # Reducido de 0.1 a 0.05
            'corners_sigmoid': corners_sigmoid * 0.05   # Mantenido en 0.05
        }
        
        # Suma directa de los valores ponderados
        overall_score = (
            weights['ssim_sigmoid'] +
            weights['stroke_sigmoid'] +
            weights['curves_sigmoid'] +
            weights['lines_sigmoid'] +
            weights['thickness_sigmoid'] +
            weights['corners_sigmoid']
        )
        
        return overall_score
    
    def _sigmoid_score(self, score: float, center: float = 0.4, k: float = 10) -> float:
        # Asegurar que el score esté en rango válido
        score = np.clip(score, 0, 1)
        return 1 / (1 + math.exp(-k * (score - center)))
    
    def _apply_sigmoid_to_all_metrics(self, ssim_score: float, stroke_quality: float,
                                     curve_smoothness: float, line_straightness: float,
                                     thickness_uniformity: float, corner_quality: float) -> tuple:
        
        return (
            self._sigmoid_score(ssim_score, center=0.4, k=10),      # SSIM - moderadamente estricto
            self._sigmoid_score(stroke_quality, center=0.3, k=12),   # Stroke - más estricto
            self._sigmoid_score(curve_smoothness, center=0.4, k=10), # Curves - moderado
            self._sigmoid_score(line_straightness, center=0.3, k=12), # Lines - más estricto  
            self._sigmoid_score(thickness_uniformity, center=0.5, k=8), # Thickness - más tolerante
            self._sigmoid_score(corner_quality, center=0.5, k=8)     # Corners - más tolerante
        )
    
    def _generate_recommendations(self, ssim_score: float, stroke_quality: float,
                                 curve_smoothness: float, line_straightness: float,
                                 thickness_uniformity: float, corner_quality: float) -> List[str]:
        recommendations = []
        
        # Umbrales ajustados para valores sigmoidizados (generalmente más altos)
        if ssim_score < 0.7:
            recommendations.append("Practica trazando la forma básica.")
        
        if stroke_quality < 0.6:
            recommendations.append("Mantén el lápiz en contacto con el papel durante todo el trazo.")
        
        if curve_smoothness < 0.7:
            recommendations.append("Mejoremos las curvas. Practica movimientos circulares.")
        
        if line_straightness < 0.6:
            recommendations.append("Las líneas rectas necesitan más precisión.")
        
        if thickness_uniformity < 0.7:
            recommendations.append("Mantén presión constante en el lápiz durante todo el trazo.")
        
        if corner_quality < 0.7:
            recommendations.append("Necesitamos practicar las esquinas.")
        
        # Agregar recomendaciones positivas basadas en fortalezas
        strengths = []
        if ssim_score >= 0.8:
            strengths.append("forma general")
        if stroke_quality >= 0.7:
            strengths.append("continuidad del trazo")
        if curve_smoothness >= 0.8:
            strengths.append("suavidad de curvas")
        if line_straightness >= 0.7:
            strengths.append("rectitud de líneas")
        if thickness_uniformity >= 0.8:
            strengths.append("uniformidad del grosor")
        if corner_quality >= 0.8:
            strengths.append("definición de esquinas")
        
        if strengths:
            recommendations.append(f"¡Muy bien! Tienes excelente {', '.join(strengths)}.")
        
        if not recommendations:
            recommendations.append("¡Excelente trabajo! Tu letra tiene muy buena calidad.")
        
        return recommendations