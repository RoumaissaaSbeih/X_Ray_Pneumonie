"""
Script de prédiction pour une nouvelle image radiographique
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
import sys
import os

def predict_pneumonia(image_path, model_path='best_DenseNet_pneumonia.keras', model_type='DenseNet'):
    """
    Prédit si une image radiographique montre une pneumonie
    
    Args:
        image_path: chemin vers l'image à analyser
        model_path: chemin du modèle entraîné
        model_type: 'DenseNet' ou 'ResNet'
    
    Returns:
        result: 'PNEUMONIA' ou 'NORMAL'
        confidence: niveau de confiance (0-1)
    """
    
    # Vérifications
    if not os.path.exists(image_path):
        print(f"❌ Erreur: Image '{image_path}' non trouvée!")
        return None, None
    
    if not os.path.exists(model_path):
        print(f"❌ Erreur: Modèle '{model_path}' non trouvé!")
        print("   Veuillez d'abord exécuter main.py pour entraîner le modèle")
        return None, None
    
    # Chargement du modèle
    print(f"📂 Chargement du modèle: {model_path}")
    model = load_model(model_path)
    
    # Chargement et prétraitement de l'image
    print(f"🖼️ Analyse de l'image: {image_path}")
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prétraitement selon le modèle
    if model_type == 'ResNet':
        img_array = resnet_preprocess(img_array)
    else:
        img_array = densenet_preprocess(img_array)
    
    # Prédiction
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Interprétation
    result = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    # Affichage des résultats
    print("\n" + "="*50)
    print("🏥 RÉSULTAT DU DIAGNOSTIC")
    print("="*50)
    print(f"📊 Diagnostic: {result}")
    print(f"🎯 Confiance: {confidence:.2%}")
    print(f"📈 Score brut: {prediction:.4f} (0=Normal, 1=Pneumonie)")
    
    # Affichage de l'image avec le résultat
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    color = 'red' if result == 'PNEUMONIA' else 'green'
    plt.title(f"Diagnostic: {result}\nConfiance: {confidence:.2%}", 
              color=color, fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Ajout d'un texte explicatif
    if result == "PNEUMONIA":
        plt.text(0.5, -0.05, "⚠️ Signes de pneumonie détectés - Consultez un médecin", 
                ha='center', transform=plt.gca().transAxes, color='red', fontsize=12)
    else:
        plt.text(0.5, -0.05, "✅ Poumons normaux - Pas de signes de pneumonie", 
                ha='center', transform=plt.gca().transAxes, color='green', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return result, confidence

def test_with_sample():
    """Teste le modèle avec un échantillon du dataset"""
    # Chercher une image de test
    test_dirs = [
        'chest_xray/test/NORMAL',
        'chest_xray/test/PNEUMONIA'
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            images = [f for f in os.listdir(test_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
            if images:
                sample_path = os.path.join(test_dir, images[0])
                print(f"\n🔬 Test avec image échantillon: {sample_path}")
                predict_pneumonia(sample_path)
                return
    
    print("❌ Aucune image de test trouvée")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Utilisation: python predict.py chemin/vers/image.jpg
        predict_pneumonia(sys.argv[1])
    else:
        print("Usage: python predict.py <chemin_image>")
        print("Exemple: python predict.py ma_radio.jpg")
        print("\nOu testez avec un échantillon du dataset:")
        test_with_sample()