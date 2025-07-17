import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology
from typing import List, Tuple

class StrokeAnalyzer:
    def analyze_stroke_quality(self, img: np.ndarray) -> float:
        if np.sum(img) == 0:
            return 0.0
            
        # Calcular esqueleto para analizar la estructura del trazo
        skeleton = morphology.skeletonize(img > 0)
        
        # Métricas de calidad del trazo
        connectivity = self._calculate_connectivity(skeleton)
        completeness = self._calculate_completeness(img)
        
        # Asegurar que ambos valores están en rango [0,1]
        connectivity = np.clip(connectivity, 0, 1)
        completeness = np.clip(completeness, 0, 1)
        
        return (connectivity + completeness) / 2
    
    def analyze_thickness_uniformity(self, img: np.ndarray) -> float:
        if np.sum(img) == 0:
            return 0.0
        
        # Método 1: Análisis basado en esqueleto y radios locales
        skeleton_score = self._analyze_thickness_via_skeleton(img)
        
        # Método 2: Análisis basado en perfiles de grosor
        profile_score = self._analyze_thickness_via_profiles(img)
        
        # Método 3: Análisis estadístico de áreas locales
        area_score = self._analyze_thickness_via_areas(img)
        
        # Combinar los tres métodos para mayor robustez
        final_score = (skeleton_score * 0.5 + profile_score * 0.3 + area_score * 0.2)
        
        # Debug
        print(f"    Thickness Debug - Skeleton: {skeleton_score:.3f}, Profile: {profile_score:.3f}, Area: {area_score:.3f}")
        
        return np.clip(final_score, 0, 1)
    
    def analyze_curve_smoothness(self, img: np.ndarray) -> float:
        if np.sum(img) == 0:
            return 0.0
            
        # Obtener contornos con mayor precisión
        contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return 0.5  # Score neutro si no hay contornos
        
        smoothness_scores = []
        for contour in contours:
            if len(contour) < 20:  # Necesitamos suficientes puntos para analizar
                continue
            
            # Suavizar el contorno ligeramente para reducir ruido
            epsilon = 0.005 * cv2.arcLength(contour, True)
            smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(smoothed_contour) < 10:
                continue
            
            # Calcular curvatura local
            curvatures = self._calculate_curvature(smoothed_contour)
            
            # Suavidad basada en variación de curvatura
            if len(curvatures) > 0:
                # Normalizar curvaturas para evitar valores extremos
                curvatures = np.array(curvatures)
                curvature_std = np.std(curvatures)
                
                # Convertir desviación estándar a score de suavidad
                # Menor variación = mayor suavidad
                smoothness = 1 / (1 + curvature_std * 2)
                smoothness_scores.append(smoothness)
        
        return np.mean(smoothness_scores) if smoothness_scores else 0.5
    
    def analyze_line_straightness(self, img: np.ndarray) -> float:
        if np.sum(img) == 0:
            return 0.0
            
        # Mejorar detección de bordes
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)
        edges = cv2.Canny(img_blur, 30, 100, apertureSize=3)
        
        # Detectar líneas usando transformada de Hough con parámetros más sensibles
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=15, 
                               minLineLength=8, maxLineGap=3)
        
        if lines is None or len(lines) == 0:
            # Si no hay líneas detectadas, evaluar la rectitud general del contorno
            return self._evaluate_contour_straightness(img)
        
        straightness_scores = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if length < 5:  # Líneas muy cortas no son confiables
                continue
            
            # Analizar qué tan recta es la línea comparando con el contorno real
            straightness = self._measure_line_deviation(img, (x1, y1), (x2, y2))
            straightness_scores.append(straightness)
        
        return np.mean(straightness_scores) if straightness_scores else 0.3
    
    def analyze_corner_quality(self, img: np.ndarray) -> float:
        # Detectar esquinas usando Harris corner detection
        img_float = np.float32(img)
        corners = cv2.cornerHarris(img_float, 2, 3, 0.04)
        
        # Analizar la nitidez de las esquinas
        corner_quality = self._evaluate_corner_sharpness(corners, img)
        
        return corner_quality
    
    def _calculate_connectivity(self, skeleton: np.ndarray) -> float:
        """Calcula qué tan conectado está el trazo"""
        labeled, num_components = ndimage.label(skeleton)
        
        # Ideal: una sola componente conectada
        if num_components <= 1:
            return 1.0
        else:
            return max(0, 1 - (num_components - 1) * 0.2)
    
    def _calculate_completeness(self, img: np.ndarray) -> float:
        if np.sum(img) == 0:
            return 0.0
            
        # Buscar huecos en el trazo
        filled = ndimage.binary_fill_holes(img > 0)
        holes = filled.astype(int) - (img > 0).astype(int)
        
        # Asegurar que holes no sea negativo
        holes = np.maximum(holes, 0)
        
        hole_ratio = np.sum(holes) / (np.sum(img > 0) + 1e-8)
        
        # Limitar el impacto y asegurar rango [0,1]
        completeness = 1 - np.clip(hole_ratio, 0, 1)
        return max(0.0, min(1.0, completeness))
    
    def _calculate_curvature(self, contour: np.ndarray) -> List[float]:
        """Calcula la curvatura local de un contorno"""
        contour = contour.reshape(-1, 2)
        curvatures = []
        
        for i in range(2, len(contour) - 2):
            # Vectores entre puntos consecutivos
            v1 = contour[i] - contour[i-1]
            v2 = contour[i+1] - contour[i]
            
            # Calcular ángulo entre vectores
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norms > 0:
                cos_angle = np.clip(dot_product / norms, -1, 1)
                angle = np.arccos(cos_angle)
                curvatures.append(angle)
        
        return curvatures
    
    def _evaluate_contour_straightness(self, img: np.ndarray) -> float:
        contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        straightness_scores = []
        for contour in contours:
            if len(contour) < 4:
                continue
                
            # Aproximar el contorno con líneas rectas
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Calcular qué tan bien se aproxima con líneas rectas
            original_area = cv2.contourArea(contour)
            approx_area = cv2.contourArea(approx)
            
            if original_area > 0:
                similarity = min(approx_area / original_area, original_area / approx_area)
                straightness_scores.append(similarity)
        
        return np.mean(straightness_scores) if straightness_scores else 0.3
    
    def _measure_line_deviation(self, img: np.ndarray, start: Tuple[int, int], 
                               end: Tuple[int, int]) -> float:
        # Crear línea ideal con grosor similar al trazo
        line_img = np.zeros_like(img)
        cv2.line(line_img, start, end, 255, 2)
        
        # Expandir la línea ideal para ser más tolerante
        kernel = np.ones((3,3), np.uint8)
        line_img = cv2.dilate(line_img, kernel, iterations=1)
        
        # Calcular región de interés alrededor de la línea
        x1, y1 = start
        x2, y2 = end
        
        # Crear máscara para la región cercana a la línea
        mask = np.zeros_like(img)
        cv2.line(mask, start, end, 255, 10)  # Línea más gruesa para la máscara
        
        # Obtener píxeles del trazo en la región de la línea
        stroke_in_region = np.logical_and(img > 0, mask > 0)
        line_pixels = np.logical_and(line_img > 0, mask > 0)
        
        if np.sum(stroke_in_region) == 0 or np.sum(line_pixels) == 0:
            return 0.0
        
        # Calcular intersección y unión
        intersection = np.logical_and(stroke_in_region, line_pixels)
        union = np.logical_or(stroke_in_region, line_pixels)
        
        iou = np.sum(intersection) / (np.sum(union) + 1e-8)
        return iou
    
    def _evaluate_corner_sharpness(self, corners: np.ndarray, img: np.ndarray) -> float:
        # Normalizar corners
        corners_normalized = cv2.normalize(corners, None, 0, 1, cv2.NORM_MINMAX)
        
        # Contar esquinas significativas
        significant_corners = corners_normalized > 0.1
        num_corners = np.sum(significant_corners)
        
        if num_corners == 0:
            return 0.5  # Score neutro
        
        # Evaluar la nitidez promedio
        corner_values = corners_normalized[significant_corners]
        average_sharpness = np.mean(corner_values)
        
        return min(1.0, average_sharpness * 2)
    
    def _analyze_thickness_via_skeleton(self, img: np.ndarray) -> float:
        try:
            # Crear esqueleto
            skeleton = morphology.skeletonize(img > 0)
            
            if np.sum(skeleton) < 5:  # Muy pocos puntos del esqueleto
                return 0.5
            
            # Calcular mapa de distancias desde el esqueleto
            distance_map = ndimage.distance_transform_edt(img > 0)
            
            # Obtener radios en puntos del esqueleto
            skeleton_coords = np.where(skeleton)
            radii = distance_map[skeleton_coords]
            
            if len(radii) < 3:
                return 0.5
            
            # Filtrar valores extremos (ruido)
            radii = radii[radii > 0.5]  # Filtrar radios muy pequeños
            
            if len(radii) < 3:
                return 0.5
            
            # Calcular uniformidad basada en coeficiente de variación
            mean_radius = np.mean(radii)
            std_radius = np.std(radii)
            
            if mean_radius < 1e-6:
                return 0.5
            
            cv = std_radius / mean_radius
            
            # Convertir CV a score de uniformidad (menor CV = mayor uniformidad)
            # CV típico: 0.1-0.5 para trazos normales
            uniformity = max(0, 1 - cv * 2)
            
            return uniformity
            
        except Exception as e:
            print(f"Error en análisis de esqueleto: {e}")
            return 0.5
    
    def _analyze_thickness_via_profiles(self, img: np.ndarray) -> float:
        try:
            # Obtener contorno principal
            contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.5
            
            main_contour = max(contours, key=cv2.contourArea)
            
            if len(main_contour) < 10:
                return 0.5
            
            # Simplificar contorno
            epsilon = 0.01 * cv2.arcLength(main_contour, True)
            simplified = cv2.approxPolyDP(main_contour, epsilon, True)
            
            widths = []
            
            # Medir grosor en varios puntos del contorno
            for i in range(0, len(simplified), max(1, len(simplified)//10)):
                point = simplified[i][0]
                
                # Calcular dirección perpendicular aproximada
                if i < len(simplified) - 1:
                    next_point = simplified[i+1][0]
                    direction = next_point - point
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        # Perpendicular
                        perpendicular = np.array([-direction[1], direction[0]])
                        
                        # Medir grosor en esta dirección
                        width = self._measure_width_at_point(img, point, perpendicular)
                        if width > 0:
                            widths.append(width)
            
            if len(widths) < 3:
                return 0.5
            
            # Calcular uniformidad
            mean_width = np.mean(widths)
            std_width = np.std(widths)
            
            if mean_width < 1e-6:
                return 0.5
            
            cv = std_width / mean_width
            uniformity = max(0, 1 - cv * 1.5)
            
            return uniformity
            
        except Exception as e:
            print(f"Error en análisis de perfiles: {e}")
            return 0.5
    
    def _analyze_thickness_via_areas(self, img: np.ndarray) -> float:
        try:
            h, w = img.shape
            
            # Dividir imagen en una grilla
            grid_size = 8
            cell_h = h // grid_size
            cell_w = w // grid_size
            
            densities = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    y1 = i * cell_h
                    y2 = min((i+1) * cell_h, h)
                    x1 = j * cell_w
                    x2 = min((j+1) * cell_w, w)
                    
                    cell = img[y1:y2, x1:x2]
                    total_pixels = cell.size
                    stroke_pixels = np.sum(cell > 0)
                    
                    if total_pixels > 0 and stroke_pixels > 0:
                        density = stroke_pixels / total_pixels
                        if density > 0.1:  # Solo considerar celdas con trazo significativo
                            densities.append(density)
            
            if len(densities) < 3:
                return 0.5
            
            # Calcular uniformidad de densidades
            mean_density = np.mean(densities)
            std_density = np.std(densities)
            
            if mean_density < 1e-6:
                return 0.5
            
            cv = std_density / mean_density
            uniformity = max(0, 1 - cv)
            
            return uniformity
            
        except Exception as e:
            print(f"Error en análisis de áreas: {e}")
            return 0.5
    
    def _measure_width_at_point(self, img: np.ndarray, point: np.ndarray, direction: np.ndarray) -> float:
        try:
            x, y = point
            dx, dy = direction * 0.5  # Paso pequeño
            
            # Buscar hacia un lado
            dist1 = 0
            curr_x, curr_y = x, y
            
            while (0 <= curr_x < img.shape[1] and 0 <= curr_y < img.shape[0] and 
                   img[int(curr_y), int(curr_x)] > 0 and dist1 < 20):
                curr_x += dx
                curr_y += dy
                dist1 += 0.5
            
            # Buscar hacia el otro lado
            dist2 = 0
            curr_x, curr_y = x, y
            
            while (0 <= curr_x < img.shape[1] and 0 <= curr_y < img.shape[0] and 
                   img[int(curr_y), int(curr_x)] > 0 and dist2 < 20):
                curr_x -= dx
                curr_y -= dy
                dist2 += 0.5
            
            return dist1 + dist2
            
        except Exception as e:
            return 0