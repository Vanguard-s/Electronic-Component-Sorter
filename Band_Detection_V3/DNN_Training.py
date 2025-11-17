import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# --- MODIFIED IMPORTS ---
from sklearn.neighbors import LocalOutlierFactor # Using LOF for outliers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def train_dnn_classifier(dataset_path='color_dataset_histogram.csv'):
    print(f"Loading HSV Histogram dataset from '{dataset_path}'...")

    try:
        data = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: {dataset_path} not found.")
        print("Please run 'collect_data_histogram.py' first.")
        sys.exit()
    if data.empty:
        print("Error: dataset is empty.")
        sys.exit()

    # --- Extract features and labels ---
    X = data.iloc[:, 0:32].values 
    y = data['color_name'].values
    print(f"Initial dataset size: {X.shape[0]} samples")

    # --- Encode class labels ---
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    class_names = list(label_encoder.classes_)

    # --- MODIFIED: Outlier removal (using LOF) ---
    print("Removing outliers using Local Outlier Factor (LOF)...")
    # n_neighbors=20 is a common default.
    # LOF finds outliers based on local density.
    lof = LocalOutlierFactor(n_neighbors=20) 
    
    # fit_predict gives -1 for outliers and 1 for inliers
    yhat = lof.fit_predict(X) 
    
    mask = yhat != -1
    X = X[mask]
    y_encoded = y_encoded[mask]
    print(f"After outlier removal: {X.shape[0]} samples")

    # --- Class Balancing with SMOTE ---
    unique, counts = np.unique(y_encoded, return_counts=True)
    min_samples = np.min(counts)
    
    k_neighbors = max(1, min(5, min_samples - 1)) 
    
    if k_neighbors < 1:
        print(f"Warning: Smallest class has {min_samples} samples. Skipping SMOTE.")
        X_bal, y_bal = X, y_encoded
    else:
        print(f"Applying SMOTE for class balancing (k_neighbors={k_neighbors})...")
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_bal, y_bal = smote.fit_resample(X, y_encoded)
        print(f"After SMOTE: {X_bal.shape[0]} samples")

    # --- Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
    )

    # --- Convert labels to one-hot ---
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

    # --- Build DNN model (Simpler + L2 Regularization) ---
    print("\nBuilding simplified model with L2 Regularization...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32,)), 
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Initial LR
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # --- Define Callbacks ---
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2,     # Reduce LR by a factor of 5
        patience=5,     # After 5 epochs of no improvement
        min_lr=0.00001, # Don't go below this
        verbose=1
    )

    # --- Train ---
    print("\nTraining DNN color classifier on Histograms...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=100, 
        batch_size=32,
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )

    # --- Evaluate ---
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nValidation Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # --- Plot training curves ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()
    
    # --- Save everything ---
    model_filename = "color_classifier_histogram.keras"
    assets_filename = "color_classifier_histogram_assets.joblib"
    
    model.save(model_filename)
    joblib.dump({"encoder": label_encoder}, assets_filename)

    print(f"\n✅ DNN color classifier trained and saved!")
    print(f"Model file: {model_filename}")
    print(f"Assets file: {assets_filename}")

if __name__ == "__main__":
    # Make sure you have the libraries:
    # pip install tensorflow pandas scikit-learn matplotlib seaborn imbalanced-learn
    train_dnn_classifier()