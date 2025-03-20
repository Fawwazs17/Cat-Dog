# ML2TF.ipynb Cell-by-Cell Explanation

## 1. Environment Setup
```python
import tensorflow as tf
import tensorflow_datasets as tfds
...
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```
**Purpose**: Imports essential libraries and checks system configuration.  
**Key Points**:
- TensorFlow (v2.x+) for deep learning operations
- TensorFlow Datasets for accessing preconfigured datasets
- Verifies GPU availability for accelerated training

## 2. Dataset Preparation
```python
dataset, info = tfds.load("cats_vs_dogs", as_supervised=True, with_info=True)
...
test_ds = test_ds.map(format_image).batch(BATCH_SIZE)
```
**Key Operations**:
- Loads 23,000 cat/dog images from TFDS
- Splits into 18,000 training and 5,000 test images
- Resizes images to 180x180 pixels and normalizes pixel values (0-1)
- Creates batched datasets (32 images/batch) for efficient processing

## 3. Model Architecture
```python
model = keras.Sequential([...])
model.compile(...)
```
**Network Structure**:
- 3 Convolutional blocks (32, 64, 128 filters) with ReLU activation
- MaxPooling layers for dimensionality reduction
- Final dense layers for classification (2 output classes)
- Adam optimizer with sparse categorical crossentropy loss

## 4. Model Training
```python
history = model.fit(...)
```
**Training Process**:
- 10 training epochs (complete passes through dataset)
- Validate performance on test set after each epoch
- Tracks accuracy/loss metrics for both training and validation

## 5. Evaluation & Inference
```python
loss, acc = model.evaluate(test_ds)
...
plt.imshow(sample_image)
...
predictions = model.predict(...)
```
**Key Features**:
- Calculates final test accuracy (typically 85-90%)
- Visualizes sample images with predictions
- Demonstrates class probability outputs (Cat: [prob], Dog: [prob])

## 6. Google Colab Integration
```python
from google.colab import drive
drive.mount('/content/drive')
...
img_path = '/content/drive/My Drive/...'
```
**Important Notes**:
- Mounts Google Drive for accessing custom images
- Example paths show Colab-specific file structure
- Users must update paths to their own image locations

## How to Use
1. Open in Google Colab using the provided badge/link
2. Run all cells sequentially (Runtime > Run All)
3. Replace sample image paths with your own in prediction cells
4. Monitor training progress through accuracy/loss metrics

**Dependencies**:  
- TensorFlow 2.x
- Matplotlib for visualization
- Google Colab environment for Drive integration
