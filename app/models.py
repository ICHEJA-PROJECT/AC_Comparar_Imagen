from dataclasses import dataclass
from typing import List

@dataclass
class AnalysisResults:
    """Estructura para almacenar todos los resultados del análisis"""
    ssim_score: float
    stroke_quality: float
    curve_smoothness: float
    line_straightness: float
    thickness_uniformity: float
    corner_quality: float
    overall_score: float
    recommendations: List[str]