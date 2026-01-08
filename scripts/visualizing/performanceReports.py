import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pathlib
import sys

def parse_args():
    """Handles all command line arguments."""
    parser = argparse.ArgumentParser(description="Compare multiple detection curve CSVs.")
    
    # Positionals: The files to compare
    parser.add_argument(
        "files", 
        nargs="+", 
        help="List of paths to validation_output_curves.csv files"
    )
    
    # Optional: The save path
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save the plot (e.g. output/comparison.png). If omitted, shows plot instead."
    )
    return parser.parse_args()

# Setup paths
SCRIPT_DIR = pathlib.Path(__file__).parent if '__file__' in locals() else pathlib.Path.cwd()
sys.path.append(str(SCRIPT_DIR.parent.parent / "src"))

def get_metrics(curve_path):
    """Parses the CSV and extracts metric arrays."""
    df_raw = pd.read_csv(curve_path, index_col=0)
    
    def parse_col(val):
        # Cleans string like "[0.1 0.2 0.3]" into a float list
        return [float(x) for x in val.strip('[]').split()]

    df = pd.DataFrame({
        "conf": parse_col(df_raw.iloc[1, 0]),
        "f1": parse_col(df_raw.iloc[1, 1]),
        "precision": parse_col(df_raw.iloc[2, 1]),
        "recall": parse_col(df_raw.iloc[3, 1])
    })
    
    fmax = df['f1'].max()
    print(f"Loaded: {pathlib.Path(curve_path).name} | Max F1: {fmax:.3f}")
    return df

def main():
    args = parse_args()

    # Plot Styling
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Generate colors based on number of files provided
    colors = plt.cm.get_cmap('tab10', len(args.files))

    for i, file_path in enumerate(args.files):
        label = pathlib.Path(file_path).parent.name 
        df = get_metrics(file_path)
        color = colors(i)

        # Left Plot: F1 Score
        ax[0].plot(df.conf, df.f1, label=f"F1: {label}", lw=2, color=color)
        
        # Right Plot: Precision-Recall
        ax[1].plot(df.recall, df.precision, label=f"P-R: {label}", lw=2, color=color)

    # Formatting Left Plot
    ax[0].set_xlabel('Confidence')
    ax[0].set_ylabel('F1 Score')
    ax[0].set_title('F1 Score vs. Confidence')
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax[0].grid(True, ls='--', alpha=0.6)

    # Formatting Right Plot
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title('Precision-Recall Curve')
    ax[1].set_xlim(0.4, 1.0)
    ax[1].set_ylim(0.3, 1.0)
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax[1].grid(True, ls='--', alpha=0.6)

    plt.tight_layout()

    # Save vs Show Logic
    if args.output:
        save_path = pathlib.Path(args.output)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Successfully saved plot to: {save_path}")
    else:
        print("No output path provided. Displaying plot...")
        plt.show()

if __name__ == "__main__":
    main()

### Action ### Command #### Example
# Compare 2 files (Show)    python scripts\visualizing\performanceReports.py path1.csv path2.csv
# Compare 4 files (Save)    python scripts\visualizing\performanceReports.py p1.csv p2.csv p3.csv p4.csv -o results.png

# Help Menu,python performanceReports.py --help   
'''
python scripts/visualizing/performanceReports.py \
    "output\validation\detect\run12-2048\run12_2048_curves.csv" \
    "output\validation\detect\run12-4096\run12_4096_curves.csv" \
    "output\validation\detect\run13-2048\run13_2048_curves.csv" \
    "output\validation\detect\run13-4096\run13_4096_curves.csv" \
    "output\validation\detect\run13-tiled\run13-tiled_curves.csv"
'''