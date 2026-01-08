import argparse
import os
import yaml
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Random Forest Classifier using pre-generated features.")
    parser.add_argument('--data_yml', required=True, help='YAML file for data configuration.')
    parser.add_argument('--n_estimators', default=100, type=int, help='Number of trees in the RF classifier.')
    parser.add_argument('--output_name', default="random_forest_training")
    return parser.parse_args()

def load_rf_data_and_labels(data_list_path):
    """Loads features (X) and labels (y) based on the path found in the YAML file."""

    # 1. Derive the actual feature and label file names from the YOLO list path
    # Example: '.../train.txt' -> '.../train_X.csv' and '.../train_y.txt'
    
    features_path = data_list_path.replace('.txt', '_X.csv')
    labels_path = data_list_path.replace('.txt', '_y.txt')

    print(f"Loading features from: {features_path}")
    print(f"Loading labels from: {labels_path}")

    # 2. Load Features (X)
    try:
        # Load the CSV without header/index and convert to NumPy array
        X = pd.read_csv(features_path, header=None, index_col=False).to_numpy()
    except FileNotFoundError:
        print(f"Error: Features CSV file not found at {features_path}.")
        return None, None
    
    # 3. Load Labels (y)
    try:
        # Load the single column of integer labels and flatten the array
        y = pd.read_csv(labels_path, header=None, index_col=False).to_numpy().flatten()
    except FileNotFoundError:
        print(f"Error: Label file not found at {labels_path}.")
        return None, None
        
    # Crucial check: Ensure feature and label alignment
    if X.shape[0] != y.shape[0]:
        print(f"Error: Mismatch in data count. Features: {X.shape[0]}, Labels: {y.shape[0]}")
        return None, None
        
    return X, y

def plot_and_save_confusion_matrix(y_true, y_pred, class_names, split_name, save_path):
    """Calculates, plots, and saves the confusion matrix."""
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the matrix to show percentages (easier to read)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, 
                                  display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, values_format=".2f", ax=ax)
    
    plt.title(f'Random Forest Confusion Matrix ({split_name})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    path = os.path.join(save_path, f'confusion_matrix_{split_name}.png')
    plt.savefig(path)
    plt.close()
    
    print(f"\nSaved confusion matrix for {split_name} to {path}")

if __name__ == '__main__':
    args = parse_arguments()
    base_path = os.path.join("output", "training", "random_forest")
    # Join the base path with the output name
    output_dir = os.path.join(base_path, args.output_name)
    # Create the directory
    os.makedirs(output_dir, exist_ok=True)
    # --- A. Load YAML Configuration ---
    data_dir = os.path.dirname(args.data_yml)
    with open(args.data_yml, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Get paths to the training and validation files
    train_label_path = os.path.join(data_dir, cfg.get('path'), cfg.get('train'))
    val_label_path = os.path.join(data_dir, cfg.get('path'), cfg.get('val'))
    test_label_path = os.path.join(data_dir, cfg.get('path'), cfg.get('test'))

    class_names = cfg.get('names', {})

    # --- B. Data Loading ---
    X_train, y_train = load_rf_data_and_labels(train_label_path)
    X_val, y_val = load_rf_data_and_labels(val_label_path)
    X_test, y_test = load_rf_data_and_labels(test_label_path)

    if X_train is None or X_val is None:
        exit()
        
    print(f"\nTraining set: {X_train.shape[0]} samples, {X_train.shape[1]} features.")
    print(f"Validation set: {X_val.shape[0]} samples, {X_val.shape[1]} features.")

    
    # --- C. Train Random Forest ---
    print(f"\nTraining Random Forest with {args.n_estimators} estimators...")
    rf_model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    print("Random Forest Training Complete. 🌲")
    
    # --- D. Evaluate ---
    y_pred = rf_model.predict(X_val)
    
    print("\n--- Validation Results ---")
    
    # Prepare class names for the final report
    target_names = [class_names.get(i, f'Class_{i}') for i in sorted(np.unique(np.concatenate((y_val, y_train))))]
    
    print(classification_report(y_val, y_pred, target_names=target_names, digits=4))

    print("\n--- Validation Results test set ---")
    y_pred_test = rf_model.predict(X_test)
    # Prepare class names for the final report
    target_names = [class_names.get(i, f'Class_{i}') for i in sorted(np.unique(np.concatenate((y_test, y_train))))]
    
    print(classification_report(y_test, y_pred_test, target_names=target_names, digits=4))

    # If you are using the full test set
    plot_and_save_confusion_matrix(y_test, y_pred_test, target_names, "test", output_dir)

    # If you also want the validation set (optional)
    plot_and_save_confusion_matrix(y_val, y_pred, target_names, "validation", output_dir)