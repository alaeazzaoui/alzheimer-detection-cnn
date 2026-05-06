# 🧠 Détection et Classification de la Maladie d'Alzheimer via CNN

> Projet de module Deep Learning — 2ème Année Cycle Ingénieur — 2025/2026  
> **AZZAOUI Alae** | Encadrant : **M. BENLAKHDER Said**

---

## 📋 Description

Ce projet développe un système de **détection et classification automatique** des stades de la maladie d'Alzheimer à partir d'images IRM cérébrales, en utilisant un modèle CNN basé sur **ResNet-50** avec Transfer Learning.

Le système classe automatiquement une IRM en **4 stades** :

| Classe | Description |
|---|---|
| 🟢 NonDemented | Cerveau sain |
| 🟡 VeryMildDemented | Démence très légère |
| 🟠 MildDemented | Démence légère |
| 🔴 ModerateDemented | Démence modérée |

---

## 🏆 Résultats

| Métrique | Valeur |
|---|---|
| Accuracy | **98%** |
| F1-Score macro | **0.98** |
| Précision moyenne | **0.99** |
| Rappel moyen | **0.98** |

---

## 🗂️ Structure du projet

```
Alzheimer/
│
├── Alzheimer.ipynb          # Notebook principal (Google Colab)
├── best_model.pth           # Meilleurs poids sauvegardés
├── README.md                # Ce fichier
│
├── alzheimer_data/
│   ├── train/
│   │   ├── MildDemented/        (200 images)
│   │   ├── ModerateDemented/    (200 images)
│   │   ├── NonDemented/         (200 images)
│   │   └── VeryMildDemented/    (200 images)
│   └── test/
│       ├── MildDemented/        (50 images)
│       ├── ModerateDemented/    (50 images)
│       ├── NonDemented/         (50 images)
│       └── VeryMildDemented/    (50 images)
│
└── outputs/
    ├── courbes_apprentissage.png
    ├── matrice_confusion.png
    └── gradcam_results.png
```

---

## ⚙️ Installation

```bash
pip install torch torchvision
pip install matplotlib scikit-learn seaborn
pip install gradio
pip install opencv-python
```

---

## 🚀 Utilisation

### 1. Cloner / ouvrir le notebook
```
Ouvrir Alzheimer.ipynb dans Google Colab
```

### 2. Télécharger le dataset
```python
kaggle datasets download -d tourist55/alzheimers-dataset-4-class-of-images
```

### 3. Exécuter dans l'ordre
```
Cellule 1 → Installation des librairies
Cellule 2 → Téléchargement du dataset
Cellule 3 → DataLoaders (train/test)
Cellule 4 → Modèle ResNet-50
Cellule 5 → Entraînement (20 époques)
Cellule 6 → Évaluation + Matrice de confusion
Cellule 7 → Grad-CAM
Cellule 8 → Interface Gradio
```

### 4. Lancer l'interface
```python
demo.launch(share=True)  # Génère un lien public
```

---

## 🏗️ Architecture du modèle

```
ResNet-50 (backbone gelé — 23.5M paramètres)
        ↓
Linear (2048 → 256)     [entraînable]
        ↓
ReLU
        ↓
Dropout (p = 0.4)
        ↓
Linear (256 → 4)        [entraînable]
        ↓
4 classes de sortie
```

**Paramètres entraînés : ~525 312 (2.3% du total)**

---

## 📊 Pipeline complet

```
IRM cérébrale
     ↓
Prétraitement (224×224, normalisation ImageNet)
     ↓
Augmentation (flip, rotation, colorjitter) — train only
     ↓
ResNet-50 Transfer Learning
     ↓
Classification 4 stades
     ↓
Grad-CAM (explicabilité)
     ↓
Interface Gradio (démo)
```

---

## 🔥 Grad-CAM

Visualisation des zones cérébrales clés utilisées par le modèle pour sa décision :
- **Rouge / Orange** → zones très importantes pour la classification
- **Bleu / Vert** → zones peu influentes

Implémenté sur la couche `model.layer4[-1]` de ResNet-50.

---

## 🖥️ Interface Gradio

L'interface permet de :
- Uploader une image IRM
- Obtenir le **diagnostic** en temps réel
- Visualiser la **heatmap Grad-CAM**
- Consulter les **probabilités** pour chaque classe

---

## 📦 Technologies utilisées

| Technologie | Version | Usage |
|---|---|---|
| Python | 3.10+ | Langage principal |
| PyTorch | 2.0+ | Deep Learning |
| TorchVision | 0.15+ | ResNet-50, transforms |
| Gradio | 3.x | Interface démo |
| OpenCV | 4.x | Heatmap Grad-CAM |
| Scikit-learn | 1.x | Métriques évaluation |
| Matplotlib | 3.x | Visualisations |
| Seaborn | 0.x | Matrice de confusion |

---

## 👨‍💻 Auteur

- **AZZAOUI Alae**

*Projet réalisé dans le cadre du module Deep Learning — 2ème Année Cycle Ingénieur — 2025/2026*
