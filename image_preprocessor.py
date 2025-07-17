import cv2
import numpy as np
from typing import Tuple

class ImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (128, 128)):
        self.target_size = target_size
    
    def preprocess(self, image_path: str) -> np.ndarray:
        img = self._load_and_denoise(image_path)
        img = self._binarize(img)
        img = self._extract_letter(img)
        img = self._resize_with_padding(img)
        return img
    
    def preprocess_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        # Convertir bytes a array numpy
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError("No se pudo decodificar la imagen")
        
        img = self._denoise(img)
        img = self._binarize(img)
        img = self._extract_letter(img)
        img = self._resize_with_padding(img)
        return img
    
    def _load_and_denoise(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {path}")
        return self._denoise(img)
    
    def _denoise(self, img: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(img, (5, 5), 0)
    
    def _binarize(self, img: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
    
    def _extract_letter(self, img: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("Advertencia: No se encontraron contornos en la imagen")
            return img
        
        # Filtrar contornos muy pequeños
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
        
        if not valid_contours:
            print("Advertencia: No se encontraron contornos significativos")
            return img
        
        # Encontrar el contorno más grande
        main_contour = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Añadir un pequeño margen
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2*margin)
        h = min(img.shape[0] - y, h + 2*margin)
        
        extracted = img[y:y+h, x:x+w]
        
        # Verificar que la extracción sea válida
        if extracted.size == 0:
            print("Advertencia: La extracción de la letra resultó vacía")
            return img
            
        return extracted
    
    def _resize_with_padding(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        max_dim = max(h, w)
        
        # Crear imagen cuadrada con padding
        square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)
        y_offset = (max_dim - h) // 2
        x_offset = (max_dim - w) // 2
        square_img[y_offset:y_offset + h, x_offset:x_offset + w] = img
        
        # Redimensionar al tamaño final
        return cv2.resize(square_img, self.target_size, interpolation=cv2.INTER_AREA)