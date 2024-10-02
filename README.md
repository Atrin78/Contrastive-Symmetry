Brain Scan Symmetry Anomaly Detection
This project uses an autoencoder to detect anomalies in brain scans by measuring the symmetry between the left and right hemispheres. The model is trained in an unsupervised manner to identify brain anomalies, such as tumors or lesions, based on asymmetries in the reconstructed scans.

Methodology
The autoencoder is trained on healthy brain scans, learning to reconstruct symmetrical images.
Asymmetry between the reconstructed hemispheres signals an anomaly.
A contrastive loss function is used to compare the left and right brain embeddings, detecting abnormal scans.
Requirements
Install the required libraries:

bash
Copy code
pip install tensorflow keras opencv-python numpy matplotlib scikit-image seaborn scipy sklearn
Usage
Prepare Dataset: Place 200 brain CT scans (100 normal, 100 abnormal) in head_ct/ as grayscale images named 001.png, 002.png, etc.
Run Training and Testing:
bash
Copy code
python train_autoencoder.py
The script trains the model on normal brain scans, then automatically tests the model on both normal and abnormal scans, outputting results like ROC curves and AUC values.
