# Covid-19 Image Classification

## Project Overview
This repository contains a COVID-19 chest X-ray classification project using a Convolutional Neural Network (CNN).
The model is designed to distinguish between:
- `COVID-19`
- `Viral Pneumonia`
- `Normal`

## Problem Statement
COVID-19 is a rapidly spreading disease that can cause lung infections and severe respiratory distress.
This project aims to support medical image analysis by classifying X-ray images into COVID-19, viral pneumonia, or normal categories.

## Data Description
The dataset uses image data saved in NumPy format and labels stored in a CSV file.

Files included:
- `CovidImages-2.npy` — image data array
- `CovidLabels-2.csv` — image labels
- `Covid_19_Image_Classification.ipynb` — Jupyter notebook with the full workflow

Data details:
- Three classes: COVID-19, Viral Pneumonia, Normal
- Images are RGB and initially loaded at `128x128x3`
- Dataset images are later resized to `64x64x3` for model training

## Key Workflow
1. Install required Python libraries
2. Load images and labels into memory
3. Explore dataset balance and visualize sample images
4. Resize and preprocess images
5. Split the dataset into training, validation, and test sets
6. Encode target labels with one-hot encoding
7. Build, train, and evaluate a CNN model in the notebook

## Getting Started
Open `Covid_19_Image_Classification.ipynb` to run the full notebook.

### Recommended environment
Install:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow`
- `opencv-python`

## Notes
This repository contains the notebook and data files required to reproduce the analysis.
