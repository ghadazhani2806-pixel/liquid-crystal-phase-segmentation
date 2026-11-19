# Deep Learning for Liquid Crystal Phase Recognition and Texture Segmentation

## Project Overview

This project explores the use of deep learning and computer vision to recognize and segment liquid crystal phases from polarized-light microscopy images.

The objective is to identify two phases in images of liquid crystal 9BA4:

- **SmC:** Smectic C phase (background).
- **Cr:** Crystalline phase (target segmentation class).

The task is formulated as a binary semantic segmentation problem, where each pixel is classified as either SmC or Cr.

## Models

Two deep learning architectures are implemented and compared:

- **U-Net:** A convolutional encoder-decoder architecture designed for image segmentation.
- **ResNet18-U-Net:** A U-Net architecture using a ResNet18 encoder initialized with pretrained ImageNet weights.

The comparison considers segmentation quality, computational cost, and inference speed.

## Dataset and Preprocessing

The project uses polarized-light microscopy images of liquid crystal 9BA4.

The dataset includes synthetic composite images in which crystalline texture fragments are overlaid onto SmC textures. Consequently, the evaluation should be interpreted in the context of this dataset and its synthetic image-generation process.

Preprocessing includes:
- Image resizing to 256 × 256 pixels.
- RGB image input.
- Binary mask conversion.
- Image normalization.
- Data augmentation through flips, rotations, and brightness variations.

Dataset split:
- Training: 16,200 image-mask pairs.
- Internal validation: 1,799 pairs.
- Official validation: 2,000 pairs.

## Experimental Results

Both models were trained for 8 epochs using the same initial learning rate (3 × 10⁻⁴).

### Official Validation Results

| Metric (Cr class) | U-Net | ResNet18-U-Net |
|---|---:|---:|
| Dice | 0.998219 | 0.997632 |
| IoU | 0.996444 | 0.995275 |
| Precision | 0.998110 | 0.998897 |
| Recall | 0.998327 | 0.996370 |
| Training time | 2031.69 s | 1757.25 s |
| Inference time / image | 11.00 ms | 9.27 ms |
| Trainable parameters | 7.76 M | 14.03 M |

### Initial Interpretation

U-Net achieved higher Dice and IoU scores, as well as higher recall, while using fewer trainable parameters.

ResNet18-U-Net achieved higher precision and a shorter measured inference time.

These results show that both architectures perform very well on the official validation split. However, further testing on naturally occurring crystalline boundaries would be necessary to assess generalization beyond the synthetic composites.

## Project Structure

```text
liquid-crystal-phase-segmentation/
│
├── notebooks/
│   └── README.md
│
├── app/
│   └── README
