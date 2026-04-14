# COVID-19 Image Classification

## Project Overview
This project uses a Convolutional Neural Network (CNN) to classify X-ray images into three categories:
- `COVID-19`
- `Viral Pneumonia`
- `Normal`

The goal is to assist medical image analysis by distinguishing COVID-19 cases from other respiratory conditions and healthy lungs.

## Dataset
The notebook uses the following files:
- `CovidImages-2.npy` — image data stored as NumPy arrays
- `CovidLabels-2.csv` — labels for each image

The dataset contains RGB chest X-ray images resized to `128x128x3` and later reduced to `64x64x3` for model training.

## Key Steps
1. Install required libraries:
   - `numpy`
   - `pandas`
   - `seaborn`
   - `tensorflow`
   - `scikit-learn`
   - `matplotlib`
   - `opencv-python`

2. Load data from NumPy and CSV files.
3. Explore dataset balance and visualize sample images.
4. Resize images from `128x128` to `64x64` for computational efficiency.
5. Apply Gaussian blur for denoising and evaluate whether it helps model performance.
6. Split the dataset into training, validation, and test sets:
   - 80% training
   - 10% validation
   - 10% test
7. Encode labels using one-hot encoding with `LabelBinarizer`.

## Notebook
Open `Covid_19_Image_Classification.ipynb` to review the full implementation, including data loading, preprocessing, model building, and evaluation.

## Notes
- The current workspace is not initialized as a git repository, so this README file was created locally.
- To push this content to GitHub, initialize a git repository in this folder, add the file, commit, and push to the target repository.

### Suggested Git commands
```bash
git init
git add README.md
git commit -m "Add project README"
git remote add origin https://github.com/okothamollo/Covid-19-Image-Classification.git
git branch -M main
git push -u origin main
```
