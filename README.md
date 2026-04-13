# Semantic Image Segmentation

Trains and evaluates two segmentation models - a custom U-Net and a pretrained DeepLabV3+ to classify pixels as Bird, Bottle or Car (or background). Both models are trained and tested on Open Images V7.


## Requirements

Designed to be ran in Google colab environment on GPU
```bash
pip install torch torchvision numpy Pillow matplotlib scikit-learn tqdm fiftyone segmentation-models-pytorch
```

## Models

**Custom U-Net** — trained from scratch, 4 encoder/decoder blocks with skip connections.

**DeepLabV3+** — pretrained ResNet50 encoder (ImageNet weights) fine-tuned for 4 classes.

## Pipeline

### Training
- 5000 images downloaded from Open Images V7 train split (Bird, Bottle, Car)
- Split 80/20 into train and validation sets
- CrossEntropyLoss with class weights `[0.15, 1.0, 1.0, 1.0]` to reduce background dominance
- U-Net: Adam, lr=0.001 — DeepLabV3+: Adam, lr=0.0001
- Up to 10 epochs with early stopping on loss divergence
- Best model saved

### Testing
- 100 images from Open Images V7 validation split
- Pixel-level evaluation excluding background
- Model performance evaluated and compared using Accuracy, Precision, Recall metrics with per class breakdown
