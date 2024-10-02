# Brain Scan Symmetry Anomaly Detection

This project uses an **autoencoder** to detect anomalies in brain scans by measuring the symmetry between the left and right hemispheres. The model is trained in an **unsupervised manner** to identify brain anomalies, such as tumors or lesions, based on asymmetries in the reconstructed scans.

## Methodology

- The autoencoder is trained on healthy brain scans, learning to reconstruct symmetrical images.
- Asymmetry between the reconstructed hemispheres signals an anomaly.
- A **contrastive loss** function is used to compare the left and right brain embeddings, detecting abnormal scans.

## Requirements

Install the required libraries:

```bash
pip install tensorflow keras opencv-python numpy matplotlib scikit-image seaborn scipy sklearn

## Usage

1. **Prepare Dataset**: 
   - Place 200 brain CT scans (100 normal, 100 abnormal) in the `head_ct/` folder.
   - The images should be grayscale and named `001.png`, `002.png`, etc.

2. **Run Training and Testing**:
   - Train the autoencoder and test it by running:
     ```bash
     python train_autoencoder.py
     ```
   - This script trains the model on normal brain scans, then automatically tests the model on both normal and abnormal scans, producing results like ROC curves and AUC values.
