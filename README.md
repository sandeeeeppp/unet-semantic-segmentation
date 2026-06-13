# Semantic Image Segmentation using U-Net

A PyTorch implementation of a U-Net model trained to segment images from the Oxford-IIIT Pet Dataset. 

I built this for my AI/ML lab to run ablation studies on the U-Net architecture. The goal was to test how specific changes—like removing skip connections, adding batch normalization and dropout, or switching from Cross-Entropy to a combined Dice Loss—affect the model's ability to predict minority classes (specifically, the thin borders around the pets).

## Dataset
* **Source:** Oxford-IIIT Pet Dataset (7,349 images).
* **Task:** 3-class segmentation. The pixels are classified as Pet (foreground), Background, or Border.

## Results
I trained four different model configurations for 30 epochs each on a Kaggle T4 GPU. The best performing setup used a combined Cross-Entropy + Dice Loss, Batch Normalization, and a 0.3 dropout rate at the bottleneck.

Final test set metrics for the optimized model:
* **Mean IoU:** 0.7516
* **Dice Coefficient:** 0.8483
* **Border Class IoU:** 0.5340 (a significant improvement from the 0.4580 baseline)

## Tech Stack & Details
* **Framework:** PyTorch, Torchvision
* **Model:** U-Net (31.03M parameters)
* **Training:** Used Automatic Mixed Precision (AMP) and the PyTorch DataLoader with persistent workers to speed up training times.

## Files in this repository
* `unet_segmentation.ipynb`: The complete notebook containing the data preprocessing, model definitions, training loops, and ablation experiments.
* `unet_segmantation_report.pdf`: My final lab report, which includes a breakdown of the math, layer-wise parameter counts, loss curves, and qualitative visual results.
