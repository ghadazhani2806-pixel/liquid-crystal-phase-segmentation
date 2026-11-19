# Reconnaissance et segmentation des phases des cristaux liquides par apprentissage profond

*Deep Learning for Liquid Crystal Phase Recognition and Texture Segmentation*

## 1. Présentation du projet

Ce projet porte sur l'application de l'apprentissage profond et de la vision par ordinateur à l'analyse d'images microscopiques de cristaux liquides observées sous lumière polarisée.

L'objectif est de réaliser une segmentation sémantique permettant de distinguer deux phases du cristal liquide 9BA4 :

- **SmC :** phase smectique C, considérée comme l'arrière-plan.
- **Cr :** phase cristalline, correspondant à la classe d'intérêt.

Le problème est formulé comme une tâche de segmentation binaire, dans laquelle chaque pixel de l'image est associé à l'une des deux classes.

## 2. Architectures étudiées

Deux architectures de Deep Learning sont implémentées et comparées :

- **U-Net :** architecture convolutive de type encodeur-décodeur, conçue pour la segmentation d'images.
- **ResNet18-U-Net :** architecture U-Net utilisant un encodeur ResNet18 initialisé avec des poids préentraînés sur ImageNet.

La comparaison porte sur la qualité de segmentation, le coût computationnel et la rapidité d'inférence.

## 3. Jeu de données et prétraitement

Le projet utilise des images microscopiques de cristaux liquides 9BA4 observées sous lumière polarisée.

Le jeu de données comprend des images composites synthétiques, obtenues en superposant des fragments de textures cristallines sur des textures SmC. Cette particularité est prise en compte dans l'interprétation des résultats.

Les principales étapes de prétraitement sont :

- Redimensionnement des images en 256 × 256 pixels.
- Utilisation d'images RGB.
- Conversion des masques en segmentation binaire.
- Normalisation des images.
- Augmentation des données par retournements, rotations et variations de luminosité.

### Répartition des données

| Ensemble | Nombre de paires image-masque |
|---|---:|
| Entraînement | 16 200 |
| Validation interne | 1 799 |
| Validation officielle | 2 000 |

## 4. Résultats expérimentaux

Les deux architectures ont été entraînées pendant 8 époques avec un taux d'apprentissage initial identique de 3 × 10⁻⁴.

### Évaluation sur l'ensemble de validation officielle

Les métriques suivantes concernent la classe cristalline Cr.

| Métrique | U-Net | ResNet18-U-Net |
|---|---:|---:|
| Dice | 0,998219 | 0,997632 |
| IoU | 0,996444 | 0,995275 |
| Précision | 0,998110 | 0,998897 |
| Rappel | 0,998327 | 0,996370 |
| Temps d'entraînement | 2031,69 s | 1757,25 s |
| Temps d'inférence / image | 11,00 ms | 9,27 ms |
| Paramètres entraînables | 7,76 M | 14,03 M |

### Interprétation

Les résultats montrent que les deux modèles obtiennent des performances très élevées sur l'ensemble de validation officielle.

U-Net présente de meilleurs scores Dice, IoU et rappel, tout en utilisant moins de paramètres entraînables.

ResNet18-U-Net obtient une meilleure précision et un temps d'inférence mesuré plus court.

Ces résultats doivent toutefois être interprétés en tenant compte de la nature synthétique des images composites. Des évaluations complémentaires sur des frontières cristallines naturellement observées seraient nécessaires pour étudier la généralisation des modèles à des situations expérimentales plus variées.

## 5. Structure du projet

```text
liquid-crystal-phase-segmentation/
│
├── notebooks/
│   └── README.md
│
├── app/
│   └── README.md
│
├── results/
│   └── README.md
│
├── docs/
│   └── README.md
│
└── README.md
```

## 6. Technologies utilisées

- Python
- PyTorch et Torchvision
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Gradio
- Google Colab

## 7. Application interactive

Une application interactive développée avec Gradio permet de charger une image microscopique, de sélectionner l'un des deux modèles et de visualiser la segmentation prédite de la phase cristalline.

Le code source de cette application sera disponible dans le dossier `app/`.

## 8. État d'avancement

Les deux architectures ont été entraînées et évaluées. Les modèles sauvegardés, les résultats expérimentaux et le notebook Google Colab sont conservés sur Google Drive.

Le dépôt GitHub est destiné à centraliser le code source, la documentation et les résultats du projet.

## 9. Auteure

**Ghada Zhani**  
Étudiante en deuxième année du cycle ingénieur en Génie Logiciel  
ISSAT Sousse – Université de Sousse
