"""
Diagnostic médical par rayons X - Détection de pneumonie
Utilisation de DenseNet/ResNet pour classifier les images radiographiques
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import kagglehub

print("="*60)
print("🏥 SYSTÈME DE DIAGNOSTIC PAR RAYONS X - PNEUMONIE")
print("="*60)

# CONFIGURATION
CONFIG = {
    'model_type': 'DenseNet',      # 'DenseNet' ou 'ResNet'
    'img_size': (224, 224),
    'batch_size': 32,
    'epochs_initial': 5,           # 5 pour test, mettre 10 pour meilleur résultat
    'epochs_fine': 5,              # 5 pour test, mettre 20 pour meilleur résultat
    'learning_rate': 0.0001
}

print(f"\n⚙️ Configuration: {CONFIG['model_type']}")
print(f"📊 Batch size: {CONFIG['batch_size']}")
print(f"🖼️ Image size: {CONFIG['img_size']}")

# ÉTAPE 1: Téléchargement du dataset
print("\n📥 ÉTAPE 1: Téléchargement du dataset...")
print("   (Cela peut prendre quelques minutes la première fois)")

try:
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    data_dir = os.path.join(path, 'chest_xray')
    print(f"✅ Dataset téléchargé: {data_dir}")
except Exception as e:
    print(f"❌ Erreur de téléchargement: {e}")
    exit(1)

# Vérification de la structure
print(f"\n📁 Structure du dataset:")
print(f"   Train: {os.path.join(data_dir, 'train')}")
print(f"   Validation: {os.path.join(data_dir, 'val')}")
print(f"   Test: {os.path.join(data_dir, 'test')}")

# ÉTAPE 2: Préparation des générateurs de données
print("\n🔄 ÉTAPE 2: Préparation des données...")

# Choisir la fonction de prétraitement selon le modèle
if CONFIG['model_type'] == 'ResNet':
    preprocess_fn = resnet_preprocess
else:
    preprocess_fn = densenet_preprocess

# Data augmentation pour l'entraînement
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_fn,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Pas d'augmentation pour validation et test
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

# Création des générateurs
train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=CONFIG['img_size'],
    batch_size=CONFIG['batch_size'],
    class_mode='binary',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(data_dir, 'val'),
    target_size=CONFIG['img_size'],
    batch_size=CONFIG['batch_size'],
    class_mode='binary',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=CONFIG['img_size'],
    batch_size=CONFIG['batch_size'],
    class_mode='binary',
    shuffle=False
)

print(f"\n📊 Statistiques:")
print(f"   Classes: {train_generator.class_indices}")
print(f"   Train: {train_generator.samples} images")
print(f"   Validation: {val_generator.samples} images")
print(f"   Test: {test_generator.samples} images")

# ÉTAPE 3: Construction du modèle
print(f"\n🏗️ ÉTAPE 3: Construction du modèle {CONFIG['model_type']}...")

if CONFIG['model_type'] == 'ResNet':
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    # Dégeler les dernières couches
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
else:  # DenseNet
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    # Dégeler les dernières couches
    for layer in base_model.layers[-30:]:
        layer.trainable = True
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

# Compilation
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate']),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.AUC(name='auc'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]
)

print("✅ Modèle compilé avec succès!")
model.summary()

# ÉTAPE 4: Callbacks
print("\n📞 ÉTAPE 4: Configuration des callbacks...")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        f'best_{CONFIG["model_type"]}_pneumonia.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ÉTAPE 5: Entraînement
print(f"\n🚀 ÉTAPE 5: Démarrage de l'entraînement...")
print(f"   Phase 1: {CONFIG['epochs_initial']} epochs")
print("   (Entraînement des couches ajoutées uniquement)")

history1 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // CONFIG['batch_size'],
    epochs=CONFIG['epochs_initial'],
    validation_data=val_generator,
    validation_steps=val_generator.samples // CONFIG['batch_size'],
    callbacks=callbacks,
    verbose=1
)

print(f"\n🔧 Phase 2: Fine-tuning - {CONFIG['epochs_fine']} epochs")
print("   (Entraînement avec couches dégelées)")

# Dégeler plus de couches pour le fine-tuning
if CONFIG['model_type'] == 'ResNet':
    for layer in model.layers[0].layers[-50:]:
        layer.trainable = True
else:
    for layer in model.layers[0].layers[-60:]:
        layer.trainable = True

# Recompiler avec learning rate plus faible
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['learning_rate'] / 10),
    loss='binary_crossentropy',
    metrics=['accuracy', 'auc', 'precision', 'recall']
)

history2 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // CONFIG['batch_size'],
    epochs=CONFIG['epochs_fine'],
    validation_data=val_generator,
    validation_steps=val_generator.samples // CONFIG['batch_size'],
    callbacks=callbacks,
    verbose=1
)

# Fusionner les historiques
history = {}
for key in history1.history.keys():
    history[key] = history1.history[key] + history2.history[key]

# ÉTAPE 6: Évaluation
print("\n📊 ÉTAPE 6: Évaluation du modèle sur le test set...")

test_results = model.evaluate(test_generator, verbose=0)

print("\n" + "="*60)
print("📈 RÉSULTATS FINAUX")
print("="*60)

metrics = ['Loss', 'Accuracy', 'AUC', 'Precision', 'Recall']
for name, value in zip(metrics, test_results):
    print(f"   {name}: {value:.4f}")

# F1-Score
f1 = 2 * (test_results[3] * test_results[4]) / (test_results[3] + test_results[4])
print(f"   F1-Score: {f1:.4f}")

# Prédictions sur le test set
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int)
true_classes = test_generator.classes

# Matrice de confusion
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Pneumonia'],
            yticklabels=['Normal', 'Pneumonia'])
plt.title(f'Matrice de confusion - {CONFIG["model_type"]}')
plt.ylabel('Vérité terrain')
plt.xlabel('Prédiction')
plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.show()

# Rapport de classification
print("\n📋 Rapport de classification détaillé:")
print(classification_report(true_classes, predicted_classes,
                          target_names=['Normal', 'Pneumonia']))

# Courbe ROC
fpr, tpr, _ = roc_curve(true_classes, predictions)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'Courbe ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title(f'Courbe ROC - {CONFIG["model_type"]}')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png', dpi=100, bbox_inches='tight')
plt.show()

# Courbes d'apprentissage
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history['accuracy'], label='Train')
axes[0].plot(history['val_accuracy'], label='Validation')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

axes[1].plot(history['loss'], label='Train')
axes[1].plot(history['val_loss'], label='Validation')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

axes[2].plot(history['auc'], label='Train')
axes[2].plot(history['val_auc'], label='Validation')
axes[2].set_title('AUC')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('AUC')
axes[2].legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n✅ Entraînement terminé avec succès!")
print(f"💾 Modèle sauvegardé: best_{CONFIG['model_type']}_pneumonia.keras")
print("\n" + "="*60)