"""
Script pour tester rapidement le modèle après entraînement
"""

import tensorflow as tf
import os

print("🧪 TEST RAPIDE DU MODÈLE")
print("="*40)

# Vérifier si le modèle existe
model_files = [f for f in os.listdir('.') if f.endswith('.keras')]

if not model_files:
    print("❌ Aucun modèle trouvé!")
    print("   Veuillez d'abord exécuter: python main.py")
    exit(1)

print(f"✅ Modèles trouvés: {model_files}")

# Tester chaque modèle
for model_file in model_files:
    print(f"\n📂 Test de {model_file}...")
    try:
        model = tf.keras.models.load_model(model_file)
        print(f"   ✅ Modèle chargé avec succès")
        print(f"   📊 Input shape: {model.input_shape}")
        print(f"   📊 Output shape: {model.output_shape}")
        print(f"   📊 Nombre de paramètres: {model.count_params():,}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

print("\n" + "="*40)
print("✅ Vérification terminée!")