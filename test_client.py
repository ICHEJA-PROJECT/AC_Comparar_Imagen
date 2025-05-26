import requests
import sys

def test_image_comparison(original_path, compare_path):
    url = "http://localhost:5000/image/compare"
    
    try:
        with open(original_path, 'rb') as f1, open(compare_path, 'rb') as f2:
            files = {
                'original_img': ('original.jpg', f1),
                'compare_img': ('compare.jpg', f2)
            }
            
            response = requests.post(url, files=files)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Comparación exitosa!")
                print(f"Precisión: {result['precision']}")
                print(f"Porcentaje: {result['precision'] * 100:.2f}%")
                return result['precision']
            else:
                print(f"Error: {response.status_code}")
                print(response.json())
                return None
                
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python test_client.py <imagen_original> <imagen_comparar>")
        sys.exit(1)
    
    precision = test_image_comparison(sys.argv[1], sys.argv[2])
    
    if precision is not None:
        if precision >= 0.9:
            print("Excelente similitud!")
        elif precision >= 0.7:
            print("Buena similitud")
        elif precision >= 0.5:
            print("Similitud moderada")
        else:
            print("Baja similitud")