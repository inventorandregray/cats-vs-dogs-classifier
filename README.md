#  Simple Cats vs Dogs AI Image Classifier

A simple **K-Nearest Neighbors (KNN)** based image classifier to distinguish between **cats** and **dogs**.  
Built with **Python, scikit-learn, NumPy, and Pillow**, the project demonstrates a clean ML pipeline including preprocessing, PCA (optional), hyperparameter tuning, evaluation, and model persistence.

---

##  Features
- **Flexible Data Loading**
  - Load from `.npy` arrays (`images.npy`, `labels.npy`).
  - Or directly from a folder (`data/cat/*.jpg`, `data/dog/*.jpg`).
- **Preprocessing**
  - Resizing, flattening, normalization to `[0,1]`.
  - Standardization with `StandardScaler`.
- **Dimensionality Reduction**
  - Optional **PCA** (Principal Component Analysis) when features > threshold.
- **Model Training**
  - Uses **K-Nearest Neighbors (KNN)**.
  - Grid search with cross-validation for hyperparameter tuning.
- **Evaluation**
  - Reports accuracy, precision, recall, F1-score.
  - Confusion matrix + sample predictions.
- **Model Persistence**
  - Saves trained pipeline + label encoder (`.joblib`).

---

##  Command Line Arguments
| Argument           | Description                                             |
| ------------------ | ------------------------------------------------------- |
| `--npy`            | Path to `.npy` images file                              |
| `--labels`         | Path to labels file (`.npy` or `.txt`)                  |
| `--data-dir`       | Directory structured as `data_dir/<label>/*.jpg`        |
| `--out-model`      | Output filename (default: `knn_cats_dogs.joblib`)       |
| `--test-size`      | Test split size (default: 0.2)                          |
| `--random-state`   | Random seed (default: 42)                               |
| `--use-pca`        | Force PCA usage                                         |
| `--pca-threshold`  | Auto-enable PCA if features > threshold (default: 2000) |
| `--pca-components` | PCA components to try (e.g., 50 100 150)                |
| `--no-grid`        | Disable grid search (faster training)                   |
| `--cv`             | Cross-validation folds (default: 3)                     |
| `--n-jobs`         | Number of parallel jobs (default: -1)                   |
| `--resize`         | Resize images (e.g., `--resize 128 128`)                |
| `--verbose`        | Enable verbose logging                                  |

## Notes

- KNN is simple and interpretable, but not state-of-the-art for images. For large-scale tasks, use CNNs.
- PCA reduces dimensionality and speeds up training.
- KNN inference scales poorly with dataset size; consider FAISS, Annoy, or deep learning for large datasets.
- This project is suitable for educational purposes or small datasets.

