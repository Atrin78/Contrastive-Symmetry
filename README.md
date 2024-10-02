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
