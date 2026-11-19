# Application de segmentation des phases des cristaux liquides

Cette application permet de segmenter les images de microscopie de cristaux liquides afin d'identifier deux phases :

- **SmC** : phase de fond (classe 0).
- **Cr** : phase cristalline (classe 1).

L'application utilise deux modèles de Deep Learning :
- U-Net
- ResNet18-U-Net

## Fonctionnalités

- Chargement d'une image de microscopie.
- Sélection du modèle de segmentation.
- Génération du masque prédit.
- Visualisation de la superposition du masque sur l'image.
- Estimation de la proportion de la phase cristalline Cr.

## Technologies utilisées

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow
- Gradio

## Installation

Installer les dépendances depuis la racine du projet :

```bash
pip install -r requirements.txt
```

## Préparation des modèles

Les poids des modèles entraînés ne sont pas inclus dans ce dépôt GitHub.

Créer un dossier `checkpoints/` à la racine du projet et y placer les deux fichiers :

```text
checkpoints/
├── unet_baseline_best.pth
└── resnet18_unet_best.pth
```

## Exécution

Depuis la racine du projet, lancer :

```bash
python app/app.py
```

L'application sera accessible depuis l'adresse locale indiquée dans le terminal.

Un GPU compatible CUDA peut accélérer les prédictions, mais le CPU est également pris en charge.

## Remarque scientifique

Les prédictions sont destinées à l'analyse exploratoire des textures des cristaux liquides. Les performances observées sur le jeu de validation ne garantissent pas une généralisation à toutes les images expérimentales.

## Projet

**Reconnaissance et segmentation des phases des cristaux liquides par apprentissage profond**

ISSAT Sousse – Département Informatique
