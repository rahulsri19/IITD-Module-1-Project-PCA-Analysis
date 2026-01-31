# MNIST PCA Analysis

**Author:** Rahul Srivastava

## Introduction

The MNIST dataset is a comprehensive collection of handwritten digits ranging from 0 to 9. It includes 60,000 grayscale images for training and 10,000 images for testing. Each image is 28 by 28 pixels, with pixel values ranging from 0 to 255. In this project, we focus on analyzing a subset of the test dataset and apply Principal Component Analysis (PCA) to reduce dimensionality.

## Tasks

### Task 1: Eigen Decomposition

Compute the eigen decomposition of the sample covariance matrix.

### Task 2: PCA for Dimensionality Reduction

Apply PCA via eigen decomposition to reduce the dimensionality of the images for each p ∈ {50, 250, 500}.

### Task 3: Data Reconstruction

Using the reduced data from Task 2, reconstruct the original images. Use the property of orthonormal matrices for reconstruction.

### Task 4: Error Comparison (PSNR)

Compare the error between the original and reconstructed images for 5 randomly selected images using Peak Signal-to-Noise Ratio (PSNR).

## Key Results

### Eigenvalues
The top 5 eigenvalues from the eigen decomposition of the covariance matrix are:
- Eigenvalue 1: 4.843252
- Eigenvalue 2: 3.708616
- Eigenvalue 3: 2.873017
- Eigenvalue 4: 2.520861
- Eigenvalue 5: 2.376909

### Variance
Variance explained by top components:
----------------------------------------
Component      Individual %        Cumulative %        
----------------------------------------
First 1         9.7140              9.7140              
First 5         4.7673              32.7380             
First 10        2.2518              47.7169             
First 20        1.1534              63.5306             
First 50        0.3434              82.3426             
First 100       0.1030              91.7155             
First 200       0.0281              97.0034             
First 500       0.0010              99.9592             
----------------------------------------

### Applying PCA for dimensionality reduction...
================================================

p = 50:
Original data shape: 784
Transformed data shape: 50
Dimensionality reduction: 93.62%
Variance retained: 82.34%

p = 250:
Original data shape: 784
Transformed data shape: 250
Dimensionality reduction: 68.11%
Variance retained: 98.14%

p = 500:
Original data shape: 784
Transformed data shape: 500
Dimensionality reduction: 36.22%
Variance retained: 99.96%

### Reconstructing images from reduced data...
================================================

p = 50:
Reconstructed data shape: (4000, 784)
Data range: [-0.5705, 1.5007]

p = 250:
Reconstructed data shape: (4000, 784)
Data range: [-0.2961, 1.3592]

p = 500:
Reconstructed data shape: (4000, 784)
Data range: [-0.1669, 1.1758]

### Calculating PSNR for reconstructed images...
================================================================================
Image Index     Label          p=50           P=250           p=500          
--------------------------------------------------------------------------------
2654            6              21.01          33.25          59.20          
1865            4              21.01          32.93          57.41          
2585            9              23.11          34.29          58.77          
302             1              25.32          36.84          64.71          
2703            0              16.70          28.96          47.07          
--------------------------------------------------------------------------------

### Average PSNR across all samples:
p = 50: Average PSNR = 21.43 dB
p = 250: Average PSNR = 33.25 dB
p = 500: Average PSNR = 57.43 dB
================================================================================

These results show that reconstruction quality improves significantly with more principal components, with p=500 achieving high PSNR values indicating good image fidelity.

## Methodology

The analysis follows these key steps:

1. **Data Loading and Preprocessing**:
   - Load the MNIST test dataset using TensorFlow/Keras.
   - Select a subset of images (e.g., 1000 samples) for analysis.
   - Normalize pixel values to the range [0, 1].
   - Reshape images from (28, 28) to vectors of 784 features.

2. **Data Centering**:
   - Compute the mean of the dataset across all samples.
   - Subtract the mean from each sample to center the data around zero.

3. **Covariance Matrix Computation**:
   - Calculate the sample covariance matrix using `np.cov` with `bias=True`.

4. **Eigen Decomposition**:
   - Perform eigen decomposition on the covariance matrix using `np.linalg.eigh` (optimized for symmetric matrices).
   - Sort eigenvalues and corresponding eigenvectors in descending order.

5. **Variance Analysis**:
   - Compute the proportion of variance explained by each principal component.
   - Calculate cumulative variance to assess dimensionality reduction impact.

6. **PCA Dimensionality Reduction**:
   - For each specified number of components p ∈ {50, 250, 500}:
     - Select the top p eigenvectors.
     - Project the centered data onto the new subspace: `X_pca = X_centered @ eigenvectors[:, :p]`.

7. **Image Reconstruction**:
   - Reconstruct images from the reduced representation: `X_reconstructed = X_pca @ eigenvectors[:, :p].T + mean`.
   - Utilize the orthonormality property of eigenvectors for accurate reconstruction.

8. **Error Evaluation**:
   - Select 5 random images for comparison.
   - Compute Peak Signal-to-Noise Ratio (PSNR) for each reconstructed image:
     - Calculate Mean Squared Error (MSE) between original and reconstructed images.
     - PSNR = 20 * log10(max_pixel / sqrt(MSE)), where max_pixel = 1.0.
   - Average PSNR across samples for each p value.

9. **Visualization**:
   - Plot sample images, eigenvalue distributions, cumulative variance, and reconstructed images for qualitative assessment.

## Conclusion

This analysis demonstrates the effectiveness of Principal Component Analysis (PCA) for dimensionality reduction on the MNIST dataset. The eigen decomposition reveals that the first few principal components capture significant variance, with the top 5 eigenvalues accounting for substantial portions of the total variance.

## Dependencies

- numpy
- matplotlib
- tensorflow

## How to Run

Open the notebook `MNIST_PCA_Analysis.ipynb` in Jupyter and run the cells sequentially.

## References

1. MNIST Dataset: http://yann.lecun.com/exdb/mnist/
2. Python Peak Signal-to-Noise Ratio (PSNR): https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
3
