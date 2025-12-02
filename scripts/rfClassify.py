import argparse
import os
import yaml
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Random Forest Classifier using pre-generated features.")
    parser.add_argument('--data_yml', required=True, help='YAML file for data configuration.')
    parser.add_argument('--n_estimators', default=100, type=int, help='Number of trees in the RF classifier.')
    return parser.parse_args()

def load_data_and_labels(data_list_path, features_path):
    """Loads features from .npy file and labels from the .txt file."""
    print(f"Loading features from: {features_path}")
    print(f"Loading labels from: {data_list_path}")

    # Load Features
    try:
        X = np.load(features_path)
    except FileNotFoundError:
        print(f"Error: Features file not found at {features_path}. Please check your path.")
        return None, None
    
    # Load Labels
    try:
        with open(data_list_path, 'r') as f:
            # We only need the second element (the class ID) from each line
            labels = [int(line.strip().split(' ')[1]) for line in f if line.strip()]
        y = np.array(labels)
    except FileNotFoundError:
        print(f"Error: Label file not found at {data_list_path}. Cannot align features.")
        return None, None
        
    # Crucial check: Ensure features and labels match in count
    if X.shape[0] != y.shape[0]:
        print(f"Error: Mismatch in data count. Features: {X.shape[0]}, Labels: {y.shape[0]}")
        return None, None
        
    return X, y

if __name__ == '__main__':
    args = parse_arguments()
    
    # --- A. Load YAML Configuration ---
    data_dir = os.path.dirname(args.data_yml)
    with open(args.data_yml, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Get paths to the training and validation files
    train_label_path = os.path.join(data_dir, cfg.get('train'))
    val_label_path = os.path.join(data_dir, cfg.get('val'))
    
    # Derive feature file paths from the label file paths
    train_features_path = train_label_path.replace('.txt', '_features.npy')
    val_features_path = val_label_path.replace('.txt', '_features.npy')
    
    class_names = cfg.get('names', {})

    # --- B. Data Loading ---
    X_train, y_train = load_data_and_labels(train_label_path, train_features_path)
    X_val, y_val = load_data_and_labels(val_label_path, val_features_path)

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