import argparse
import pathlib
import pandas as pd
import numpy as np
import sys
import os

# --- Setup for custom module imports (from notebook) ---
'''
This script is for performing transect analysis including even subsampling and comparing the resultant biomass calculation.

You must run inference on the collect images first, or have a pre-computed inference assessment preditions.csv file.

Subsampled image paths are saved to a text file for further analysis and for LABELING

A plot is generated to compare the biomass densities of the full transect and the subsampled transect.

'''
# Assuming 'src' directory is one level up from the script's location
SCRIPT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT / "src"))
from transectUtils import Transects
# -----------------------------------------------------

def parse_args():
    """
    Parses command-line arguments for the transects analysis script.
    """
    parser = argparse.ArgumentParser(description="Perform transect analysis including even subsampling and biomass calculation.")
    parser.add_argument("--results_file", type=pathlib.Path, required=True,
        help="Path to the inference inference_results.csv file."
    )
    parser.add_argument("--collect_id", type=str, default=None,
        help="The specific 'collect_id' to filter the inference assessment data by. If not provided, the script will process all data in the file."
    )
    parser.add_argument("--image_dir", type=pathlib.Path, default=pathlib.Path("Z:/__Organized_Directories_InProgress/2024_UnpackedCollects"),
        help="The base directory containing the collected images. Default: Z:/__Organized_Directories_InProgress/2024_UnpackedCollects."
    )
    # Optional analysis arguments
    parser.add_argument("--subsample_fraction", type=float, default=0.30,
        help="Fraction (0.0 to 1.0) of unique images to select for even subsampling."
    )
    parser.add_argument("--weight_filter", type=float, default=80.0,
        help="Weight threshold in grams (g) to filter out anomalous detections. Detections with a weight greater than this value will be excluded from summation. Default is 80g."
    )

    return parser.parse_args()

def save_subsampled_image_paths(subsample: pd.DataFrame, image_dir: pathlib.Path, collect_id: str, output_dir: pathlib.Path, frac: float):
    """
    Constructs the absolute file paths for the subsampled images and saves them 
    to a text file in the specified output_dir.
    """
    
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure the output directory exists
    
    # Define the output file name
    output_filename = f"subsampled_images_{frac}.txt"
    output_path = output_dir / output_filename
    
    if subsample.empty:
        print(f"Warning: Subsample dataframe is empty. Skipping saving image path list.")
        return

    # Extract unique filenames from the subsample
    subsample_filenames = subsample.Filename.unique()
    
    # The full path to the image files is: image_dir / collect_id / PrimaryImages / filename
    primary_image_path = image_dir / collect_id / "PrimaryImages"
    
    # Construct the full absolute path for each subsampled filename
    path_list = []
    for filename in subsample_filenames:
        # Filename should contain the extension (e.g., 'image001.png')
        full_path = primary_image_path / filename 
        
        # Check if the file exists using the full filename from the DF
        if full_path.is_file():
            path_list.append(str(full_path.resolve()) + '\n')
        else:
            # Fallback check for missing extension (e.g., assuming .png)
            if not pathlib.Path(filename).suffix:
                 full_path_png = primary_image_path / (filename + ".png")
                 if full_path_png.is_file():
                    path_list.append(str(full_path_png.resolve()) + '\n')
            
    
    if not path_list:
        print(f"\nWarning: Could not find any subsampled image files in {primary_image_path}. Skipping file path list creation.")
        return

    # Sort the paths
    path_list.sort()
    
    try:
        with open(output_path, 'w') as f:
            f.writelines(path_list)
        print(f"Successfully saved {len(path_list)} subsampled image paths to {output_path}")
    except Exception as e:
        print(f"\nError saving subsampled image paths to file: {e}")

def main():
    """
    Main function to run the transect analysis.
    """
    args = parse_args()
    
    # Configuration based on arguments
    collect_id = args.collect_id
    image_dir = args.image_dir
    frac = args.subsample_fraction
    filt_wt = args.weight_filter
    
    # --- Define Output Folder ---
    SCRIPT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
    PROJECT_ROOT = SCRIPT_DIR.parent  # Ensure we use POSIX paths for consistency
    output_folder = PROJECT_ROOT / "output" / "transects" / collect_id
    # Ensure output folder exists before proceeding
    output_folder.mkdir(parents=True, exist_ok=True)
    # ----------------------------
    
    print(f"--- Transects Analysis Started ---")
    print(f"Loading data from: {args.results_file}")
    print(f"Filtering by collect_id: {collect_id}")
    print(f"Base image directory: {image_dir}")
    print(f"Subsampling fraction: {frac*100:0.0f}%%")
    print(f"Weight filter threshold: {filt_wt}g")
    print(f"Analysis outputs saving to: {output_folder}")

    # Load inference assessment and filter by collect_id (from cell 5)
    try:
        inference_assmt = pd.read_csv(args.results_file, index_col=0, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Inference assessment file not found at {args.results_file}")
        sys.exit(1)
        
    # determine cid column
    if 'collect_id' in inference_assmt.columns:
        cid = 'collect_id'
    elif 'CollectID' in inference_assmt.columns:
        cid = 'CollectID'
    else:
        print(f"Error: No valid collect_id column found.")
        sys.exit(1)

    inference_assmt = inference_assmt[inference_assmt[cid] == collect_id].copy()
    print(f"CollectID filtered data shape: {inference_assmt.shape}")

    if inference_assmt.empty:
        print(f"Error: No data found for collect_id '{collect_id}' in the assessment file.")
        sys.exit(1)

    print(f"\nOriginal data shape: {inference_assmt.shape}")
    
    # --- Even Subsampling Logic ---
    
    unique_filenames = inference_assmt.Filename.unique()
    num_unique_images = len(unique_filenames)
    print(f"Number of unique images: {num_unique_images}")

    n = max(1, int(frac * num_unique_images))
    step_size = num_unique_images // n 

    if step_size == 0 and n > 0:
        step_size = 1

    selected_indices = range(0, num_unique_images, step_size)
    even_subsample_images = pd.Series(unique_filenames).iloc[selected_indices].tolist()

    subsample = inference_assmt[inference_assmt.Filename.isin(even_subsample_images)].copy()

    print(f"Even Subsampling: {len(even_subsample_images)} images, total detections: {subsample.shape[0]}")
    
    # --- Check for Unpacked Subsampled Images ---
    img_directory = args.image_dir / args.collect_id / "PrimaryImages"
    
    if not img_directory.is_dir():
        print(f"\nWarning: Image directory not found at {img_directory}. Skipping subsampled image check.")
    else:
        subsample_fn_with_ext = subsample.Filename.unique()
        subsample_fn_base = set(pathlib.Path(x).stem for x in subsample_fn_with_ext)
        print(f"Subsampled filenames: {len(subsample_fn_base)} unique images (base names)")

        unpacked_filenames = os.listdir(img_directory)
        unpacked_filenames_base = set(x.split(".")[0] for x in unpacked_filenames)

        intersection_size = len(subsample_fn_base.intersection(unpacked_filenames_base))
        
        assert intersection_size == len(subsample_fn_base), \
            f"Subsampled filenames ({len(subsample_fn_base)}) do not match unpacked filenames ({intersection_size} found in intersection). Directory: {img_directory}"
        print(f"Check successful: All {len(subsample_fn_base)} subsampled images found in the unpacked directory.")
    # ---------------------------------------------------

    # --- Calculate Weights and Biomass Density ---
    
    total_infer_weight = inference_assmt[inference_assmt.box_DL_weight_g_corr <= filt_wt].box_DL_weight_g_corr.sum()
    subsample_infer_weight = subsample[subsample.box_DL_weight_g_corr <= filt_wt].box_DL_weight_g_corr.sum()

    infer_biomass_df = pd.DataFrame()
    subsample_biomass_df = pd.DataFrame()
    avg_infer_biomass = np.nan
    avg_subsample_biomass = np.nan

    try:
        infer_biomass_df = Transects.biomass_transects(inference_assmt)
        avg_infer_biomass = infer_biomass_df.biomass_g_p_m2.mean()
    except Exception as e:
        print(f"\nWarning: Could not compute full biomass density. Transects.biomass_transects failed: {e}")
        
    try:
        subsample_biomass_df = Transects.biomass_transects(subsample)
        avg_subsample_biomass = subsample_biomass_df.biomass_g_p_m2.mean()
    except Exception as e:
        print(f"Warning: Could not compute subsample biomass density. Transects.biomass_transects failed: {e}")
    
    # --- Final Output Formatting & Saving ---
    
    results_markdown = []
    
    # Header and Metadata for the append
    results_markdown.append(f"## Transects Analysis Results for `{collect_id}`")
    results_markdown.append(f"**Run Time:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results_markdown.append(f"**Subsampling Fraction:** {frac*100:0.0f}%")
    results_markdown.append(f"**Weight Filter:** < {filt_wt}g")
    results_markdown.append("")
    
    # Summary Table
    results_markdown.append("| Metric | Full Transect | Subsampled |")
    results_markdown.append("| :--- | :--- | :--- |")
    results_markdown.append(f"| Total Detections | {inference_assmt.shape[0]:,} | {subsample.shape[0]:,} |")
    results_markdown.append(f"| Unique Images | {num_unique_images:,} | {len(even_subsample_images):,} |")
    results_markdown.append(f"| Total Weight (g) | {total_infer_weight:,.2f} g | {subsample_infer_weight:,.2f} g |")
    results_markdown.append(f"| Avg Biomass Density (g/m²) | {avg_infer_biomass:,.4f} g/m² | {avg_subsample_biomass:,.4f} g/m² |")
    results_markdown.append("\n\n---\n\n") # Separator for appended runs

    # Print summary to console
    print("\n--- Results Summary ---")
    print(f"Total Inferred Weight (Filtered < {filt_wt}g): {total_infer_weight:,.2f} g")
    print(f"Subsample Inferred Weight (Filtered < {filt_wt}g): {subsample_infer_weight:,.2f} g")
    print(f"Average Inferred Biomass Density: {avg_infer_biomass:,.4f} g/m²")
    print(f"Average Subsample Biomass Density: {avg_subsample_biomass:,.4f} g/m²")

    # Save subsampled image paths 
    save_subsampled_image_paths(subsample, image_dir, collect_id, output_folder, frac)

    # Save biomass comparison plot
    if not infer_biomass_df.empty and not subsample_biomass_df.empty:
        plot_save_path = output_folder / f"transect_plot_{frac}.png"
        try:
            Transects.plot_biomass_comparison_moving_average(
                primary_transect=infer_biomass_df, 
                secondary_transect=subsample_biomass_df, 
                secondary_lbl="subsampled inference", 
                secondary_window=round(100*frac), 
                save_path = plot_save_path
            )
            print(f"Successfully saved biomass comparison plot to {plot_save_path}")
        except Exception as e:
            print(f"Error saving biomass comparison plot: {e}")
    else:
        print("Skipping biomass comparison plot: Required biomass dataframes are empty.")

    # Save results summary to Markdown file (APPEND mode)
    summary_path = output_folder / "results_summary.md"
    try:
        with open(summary_path, 'a', encoding='utf-8') as f:
            f.write('\n'.join(results_markdown))
        print(f"Successfully appended results summary to {summary_path}")
    except Exception as e:
        print(f"Error saving results summary file: {e}")

if __name__ == "__main__":
    main()