"""
Simple Cats vs Dogs classifier using scikit-learn KNN.

Supports:
 - loading images from a .npy file (images array) optionally with labels
 - loading images from a folder structured as: data_dir/<label>/*.jpg
 - preprocessing, optional PCA, hyperparameter search (GridSearchCV)
 - evaluation and joblib saving of the best pipeline + label encoder

Requirements:
    pip install numpy scikit-learn pillow joblib matplotlib
"""

from pathlib import Path
import os
import argparse
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def load_images_from_folder(folder, target_size=None, verbose=False):
    """
    folder expected format: folder/<label>/*.jpg (or png)
    returns: images (n_samples, H, W, C), labels (n_samples,)
    """
    images = []
    labels = []
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Data directory not found: {folder}")

    classes = [d for d in sorted(folder.iterdir()) if d.is_dir()]
    if not classes:
        raise ValueError(f"No class-subfolders found in {folder}. Expect folder/<label>/* images.")
    for cls in classes:
        cls_name = cls.name
        for f in sorted(cls.iterdir()):
            if not f.is_file():
                continue
            try:
                img = Image.open(f).convert('RGB')
                if target_size:
                    img = img.resize(target_size, Image.BILINEAR)
                arr = np.asarray(img)
                images.append(arr)
                labels.append(cls_name)
            except Exception as e:
                if verbose:
                    print(f"Warning: couldn't load {f}: {e}", file=sys.stderr)
    if not images:
        raise ValueError("No images loaded - check data directory and file extensions")
    return np.array(images), np.array(labels)


def load_npy_images(images_path, labels_path=None):
    images = np.load(images_path)
    if labels_path:
        labels = np.load(labels_path)
    else:
        labels = None
    return images, labels


def ensure_flatten_and_normalize(images):
    """
    Convert images to 2D array (n_samples, n_features) and normalize to [0,1]
    Accepts images already flattened.
    """
    images = np.array(images)
    if images.ndim == 2:
        X = images.astype('float32')
    else:
        n_samples = images.shape[0]
        X = images.reshape(n_samples, -1).astype('float32')
    # Normalize from [0..255] to [0..1] if max > 1
    if X.max() > 1.0:
        X /= 255.0
    return X


def build_and_train(X_train, y_train, use_pca=False, pca_components=None,
                    grid_search=True, cv=3, n_jobs=-1):
    steps = []
    steps.append(('scaler', StandardScaler()))
    if use_pca:
        pca = PCA(random_state=42)
        steps.append(('pca', pca))
    steps.append(('knn', KNeighborsClassifier()))
    pipeline = Pipeline(steps)

    param_grid = {
        'knn__n_neighbors': [3, 5, 7],
        'knn__weights': ['uniform', 'distance']
    }
    if use_pca:
        if pca_components:
            param_grid['pca__n_components'] = pca_components
        else:
            param_grid['pca__n_components'] = [50, 100, 150]

    if grid_search:
        gs = GridSearchCV(pipeline, param_grid=param_grid, cv=cv,
                          n_jobs=n_jobs, verbose=1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        return best, gs
    else:
        pipeline.set_params(**{k: v[0] for k, v in param_grid.items()
                               if isinstance(v, list)})
        pipeline.fit(X_train, y_train)
        return pipeline, None


def evaluate_model(model, X_test, y_test, label_encoder=None):
    y_pred = model.predict(X_test)
    if label_encoder is not None and hasattr(label_encoder, "inverse_transform"):
        try:
            y_test_dec = label_encoder.inverse_transform(y_test)
            y_pred_dec = label_encoder.inverse_transform(y_pred)
        except Exception:
            y_test_dec = y_test
            y_pred_dec = y_pred
    else:
        y_test_dec = y_test
        y_pred_dec = y_pred

    acc = accuracy_score(y_test_dec, y_pred_dec)
    report = classification_report(y_test_dec, y_pred_dec, zero_division=0)
    cm = confusion_matrix(y_test_dec, y_pred_dec,
                          labels=np.unique(np.concatenate([y_test_dec, y_pred_dec])))
    return acc, report, cm, y_test_dec, y_pred_dec


def main(args):
    # Load data
    if args.npy:
        images, labels = load_npy_images(args.npy, args.labels)
        if labels is None and args.data_dir:
            _, labels = load_images_from_folder(args.data_dir)
    elif args.data_dir:
        images, labels = load_images_from_folder(
            args.data_dir,
            target_size=tuple(args.resize) if args.resize else None,
            verbose=args.verbose
        )
    else:
        raise ValueError("Provide either --npy <images.npy> or --data-dir <folder>")

    if labels is None:
        raise ValueError("Labels were not provided or could not be inferred.")

    if len(images) != len(labels):
        raise ValueError(f"Number of images ({len(images)}) and labels ({len(labels)}) must match")

    # Flatten & normalize
    X = ensure_flatten_and_normalize(images)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # train/test split with stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # PCA decision
    n_features = X_train.shape[1]
    use_pca = args.use_pca or (n_features > args.pca_threshold)
    pca_components = args.pca_components if args.pca_components else None

    print(f"Samples: {X.shape[0]}, Features per sample: {n_features}, "
          f"classes: {len(le.classes_)}, use_pca={use_pca}")

    best_pipeline, gs = build_and_train(
        X_train, y_train,
        use_pca=use_pca,
        pca_components=pca_components,
        grid_search=not args.no_grid,
        cv=args.cv,
        n_jobs=args.n_jobs
    )

    if gs is not None:
        print("GridSearch best params:", gs.best_params_)

    # Evaluate
    acc, report, cm, y_test_dec, y_pred_dec = evaluate_model(best_pipeline, X_test, y_test, label_encoder=le)
    print(f"\nTest Accuracy: {acc:.4f}\n")
    print("Classification report:\n", report)
    print("Confusion matrix (rows=true, cols=pred):\n", cm)

    # Save model + label encoder
    joblib.dump({'pipeline': best_pipeline, 'label_encoder': le}, args.out_model)
    print(f"Saved pipeline + label encoder to: {args.out_model}")

    # Sample predictions
    n_show = min(10, len(y_test_dec))
    print("\nSample predictions (true => pred):")
    for i in range(n_show):
        print(f"  {y_test_dec[i]} => {y_pred_dec[i]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple KNN image classifier for cats vs dogs")
    parser.add_argument('--npy', type=str, help="Path to images .npy file")
    parser.add_argument('--labels', type=str, help="Path to labels .npy file or .txt")
    parser.add_argument('--data-dir', type=str, help="Data dir structured as data_dir/<label>/*.jpg")
    parser.add_argument('--out-model', type=str, default='knn_cats_dogs.joblib')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--use-pca', action='store_true')
    parser.add_argument('--pca-threshold', type=int, default=2000)
    parser.add_argument('--pca-components', nargs='+', type=int, default=None)
    parser.add_argument('--no-grid', action='store_true')
    parser.add_argument('--cv', type=int, default=3)
    parser.add_argument('--n-jobs', type=int, default=-1)
    parser.add_argument('--resize', nargs=2, type=int)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    main(args)

