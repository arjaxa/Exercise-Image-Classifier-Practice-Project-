# Exercise Image Classifier

A lightweight deep learning image classifier that recognizes four common exercises from images:

- Lunge
- Plank
- Push-up
- Squat

The model was built with TensorFlow/Keras using transfer learning with MobileNetV2 and deployed as a streamlit application.

## Streamlit Application

https://exclassifier.streamlit.app/

Upload an exercise image to the streamlit application and the model predicts:

- The exercise class
- Prediction confidence
- Probability for each class

## Model Architecture

The project uses MobileNetV2 pretrained on ImageNet as a feature extractor.

The classification pipeline is:

Input Image
↓
MobileNetV2
↓
Global Average Pooling
↓
Dropout
↓
Dense Softmax Classifier
↓
4 Exercise Classes

The MobileNetV2 base was initially frozen while a custom classification head was trained. The final layers were then fine-tuned using a low learning rate.

## Dataset

A small custom image dataset was used for this project.

| Split | Images |
|---|---:|
| Training | 96 |
| Validation | 24 |
| Test | 40 |
| Classes | 4 |

The four classes are:

- Lunge
- Plank
- Push-up
- Squat

Images were resized to `224 × 224` pixels and normalized to the `[0, 1]` range.

## Results

The final model achieved:

**70% test accuracy**

on the held-out 40-image test set.


## Technologies

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pillow
- Streamlit
- Jupyter Notebook

## Concepts Practiced

This project was created to reinforce concepts from the IBM AI Engineering Professional Certificate, particularly the deep learning and Keras material.

### Transfer Learning

A pretrained MobileNetV2 model was used to extract visual features from exercise images.

### Feature Extraction

The pretrained MobileNetV2 layers were initially frozen while a custom classification head was trained.

### Fine-Tuning

The final layers of MobileNetV2 were later unfrozen and fine-tuned using a small learning rate.

### Keras Functional API

The custom classifier was constructed using the Keras Functional API.

### Regularization

Dropout was added before the final classification layer to help reduce overfitting.

### Model Evaluation

Training and validation performance were monitored during training, followed by evaluation on a held-out test set.



