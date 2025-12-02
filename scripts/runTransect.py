import argparse
import pathlib
import subprocess
import sys
import os
import pandas as pd
from typing import Dict, Any
import numbers
SCRIPT_DIR = pathlib.Path(__file__).parent if '__file__' in locals() else pathlib.Path.cwd()
sys.path.append(str(SCRIPT_DIR.parent / "src"))
try:
    from reportFunctions import Reports
    from transects import Transects
except ImportError as e:
    print(f"Error importing modules: {e}. Check your PYTHONPATH and 'src' directory.")
    sys.exit(1)

# ======================================================================
# 1. CONFIGURATION AND PATHS
# ======================================================================
def arg_parser():
    """Main function to parse arguments and run the report generation pipeline."""
    parser = argparse.ArgumentParser(description="Generate a summary report for a specific transect inference run.")
    parser.add_argument("--run_directory", type=str, required=True, default="D:\\ageglio-1\\gobyfinder_yolov8\\output\\test_runs",
                        help="The name of the test run folder where results are stored.")
    parser.add_argument("--test_run", type=str, required=False, default="default_test_run",
                        help="The name of the test run folder where results are stored.")
    parser.add_argument("--collect_id", type=str, required=False, default='20200806_001_Iver3069_ABS1',
                        help="The specific collect_id (transect) to analyze.")
    parser.add_argument("--confidence_threshold", type=float, required=False, default=0.3,
                        help="The confidence threshold for choosing inference results file.")
    parser.add_argument("--run_results", action="store_true",
                        help="Flag to run the results processing step.")
    return parser.parse_args()

def load_config(run_directory: str, test_run: str, collect_id: str, confidence_threshold: float) -> Dict[str, pathlib.Path]:
    """Defines and resolves all necessary file paths."""
    
    # Use environment variables or a config file for Z: drive to improve portability.
    BASE_DRIVE = r"Z:" 
    
    PATHS = {
        "poly_lbl_assmt": BASE_DRIVE + r"\__AdvancedTechnologyBackup\07_Database\FishScaleLabelAssessment\2020-2023_assessment_confirmedfish.pkl",
        "op_table": BASE_DRIVE + r"\__AdvancedTechnologyBackup\07_Database\OP_TABLE.xlsx",
        "metadata": BASE_DRIVE + r"\__AdvancedTechnologyBackup\07_Database\MetadataCombined\all_annotated_meta_splits_filtered_20251030.csv",
        "run_folder": pathlib.Path(f"{run_directory}\\{test_run}"),
    }

    # Derived paths, created inside the run folder
    PATHS.update({
        "lbl_inference_assmt": PATHS["run_folder"] / f"inference_results_{confidence_threshold:04.2f}.csv",
        "lbl_df": PATHS["run_folder"] / "labels.csv",
        "pred_df": PATHS["run_folder"] / "predictions.csv",
        "lbl_box_results": PATHS["run_folder"] / "label_box_results.csv",
        "scores_df": PATHS["run_folder"] / "scores.csv",
        "transect_folder": PATHS["run_folder"] / f"{collect_id}_transect_summary"
    })

    # Normalize string paths to pathlib.Path for consistent handling
    for k, v in list(PATHS.items()):
        if isinstance(v, str):
            PATHS[k] = pathlib.Path(v)

    # Ensure the output directory exists
    PATHS["run_folder"].mkdir(parents=True, exist_ok=True)
    PATHS["transect_folder"].mkdir(parents=True, exist_ok=True)

    return PATHS

# ======================================================================
# 2. DATA LOADING AND FILTERING
# ======================================================================

def load_and_filter_data(paths: Dict[str, pathlib.Path], collect_id: str, confidence_threshold: float) -> Dict[str, Any]:
    """Loads and filters all necessary DataFrames for a specific transect."""
    
    print(f"Loading data for transect: {collect_id}")
    
    # 1. Load usable images for the transect (robust to column name casing)
    usable_images_df = pd.read_csv(paths["metadata"], low_memory=False)

    # Detect column names case-insensitively
    filename_col = next((c for c in usable_images_df.columns if c.lower() == "filename"), None)
    usability_col = next((c for c in usable_images_df.columns if c.lower() == "usability"), None)
    id_col = next((c for c in usable_images_df.columns if "collect_id" in c.lower() or "collectid" in c.lower()), None)

    if filename_col is None or usability_col is None or id_col is None:
        missing = [n for n, v in (("Filename", filename_col), ("Usability", usability_col), ("collect_id", id_col)) if v is None]
        raise ValueError(f"Metadata CSV is missing required column(s): {', '.join(missing)}. Available columns: {list(usable_images_df.columns)}")

    usable_images = usable_images_df[
        (usable_images_df[id_col] == collect_id) & 
        (usable_images_df[usability_col] == "Usable")
    ][filename_col].tolist()
    
    if not usable_images:
        raise ValueError(f"No usable images found for {id_col}={collect_id}.")

    # 2. Load and filter inference assessment
    infer_transects = pd.read_csv(paths["lbl_inference_assmt"], low_memory=False)
    infer_transects = infer_transects[infer_transects.Filename.isin(usable_images)].copy()
    
    if not infer_transects.empty:
        confidence_threshold = infer_transects[infer_transects.conf_pass == 1].conf.min()
        print(f"Inference results confidence threshold: {confidence_threshold:.2f}")

    # 3. Load and filter label box results
    lblres_transects = pd.read_csv(paths["lbl_box_results"], index_col=0, low_memory=False)
    lblres_transects = lblres_transects[lblres_transects.Filename.isin(usable_images)].copy()

    # 4. Load and filter polygon assessment
    poly_assmt_transects = pd.read_pickle(paths["poly_lbl_assmt"])
    poly_assmt_transects = poly_assmt_transects[poly_assmt_transects.Filename.isin(usable_images)].copy()
    
    # 5. Load and filter scores/FN data
    scores_df = pd.read_csv(paths["scores_df"], index_col=0)
    scores_df = scores_df[scores_df.Filename.isin(infer_transects.Filename.to_list())].copy()
    
    # Get IDs for performance metrics
    tpd = scores_df[scores_df.tp == 1].detect_id
    fpd = scores_df[scores_df.fp == 1].detect_id
    
    df_lbls = pd.read_csv(paths["lbl_df"], index_col=0)
    df_pred = pd.read_csv(paths["pred_df"], index_col=0)

    fndf = Reports.return_fn_df(df_lbls, df_pred, conf_thresh=confidence_threshold)
    fndf = fndf[fndf.fn==1]
    fndf.to_csv(os.path.join(paths["transect_folder"], f"false_negatives_conf_thresh_{confidence_threshold:0.2f}.csv"), index=False)
    fnid = fndf.ground_truth_id

    return {
        "infer_transects": infer_transects,
        "lblres_transects": lblres_transects,
        "poly_assmt_transects": poly_assmt_transects,
        "tpd": scores_df[scores_df.tp == 1].detect_id,
        "fpd": scores_df[scores_df.fp == 1].detect_id,
        "fnid": fnid,
        "confidence_threshold": confidence_threshold
    }

# ======================================================================
# 3. METRIC CALCULATION
# ======================================================================

def safe_sum(df: pd.DataFrame, col: str, filt_wt: float = 80) -> float:
    """Calculates sum, returning 0.0 if the df is empty or column is missing."""
    if df is not None and not df.empty and col in df.columns:
        return df[df[col] <= filt_wt][col].sum()
    return 0.0

def safe_mean_biomass(df: pd.DataFrame, Transects) -> float:
    """Calculates average biomass density, returning NaN if the df is empty."""
    if df is not None and not df.empty:
        biomass_df = Transects.biomass_transects(df)
        if not biomass_df.empty:
            return biomass_df.biomass_g_p_m2.mean()
    return float('nan')

def calculate_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculates all total weights and average biomass densities."""
    
    infer_transects = data["infer_transects"]
    lblres_transects = data["lblres_transects"]
    poly_assmt_transects = data["poly_assmt_transects"]
    tpd = data["tpd"]
    fpd = data["fpd"]
    fnid = data["fnid"]
    confidence_threshold = data["confidence_threshold"]
    
    metrics = {}

    # Safe retrieval of collect id from infer_transects (case-insensitive)
    collect_col = None
    if not infer_transects.empty:
        collect_col = next((c for c in infer_transects.columns if "collect_id" in c.lower() or "collectid" in c.lower()), None)
    metrics['Collect ID'] = infer_transects[collect_col].iloc[0] if collect_col and not infer_transects.empty else 'Unknown'
    metrics["Confidence Threshold"] = confidence_threshold if not infer_transects.empty else 'N/A'
    # --- Total Weights (g) ---
    metrics["total_lbl_box_weight"] = safe_sum(lblres_transects, 'box_DL_weight_g_corr')
    metrics["total_poly_weight"] = safe_sum(poly_assmt_transects, 'Poly_Corr_weight_g')
    metrics["total_infer_weight"] = safe_sum(infer_transects, 'box_DL_weight_g_corr')

    # --- Performance Weights (g) ---
    # True Positives: Inferred detections that matched a ground truth
    metrics["true_fish_weight"] = safe_sum(
        infer_transects[infer_transects.detect_id.isin(tpd.to_list())], 
        'box_DL_weight_g_corr'
    )
    # False Positives: Inferred detections that did not match a ground truth
    metrics["false_fish_weight"] = safe_sum(
        infer_transects[infer_transects.detect_id.isin(fpd.to_list())], 
        'box_DL_weight_g_corr'
    )
    # Missed (False Negatives): Ground truth boxes that were NOT detected
    metrics["missed_fish_weight"] = safe_sum(
        lblres_transects[lblres_transects.ground_truth_id.isin(fnid.to_list())], 
        'box_DL_weight_g_corr'
    )

    # --- Average Biomass Density (g/m^2) ---
    metrics["avg_infer_biomass"] = safe_mean_biomass(infer_transects, Transects)
    metrics["avg_lblbox_biomass"] = safe_mean_biomass(lblres_transects, Transects)
    metrics["avg_lblpoly_biomass"] = safe_mean_biomass(poly_assmt_transects, Transects)
    
    return metrics

# ======================================================================
# 4. REPORT GENERATION (Refactored to match Transects API)
# ======================================================================

def generate_report(metrics: Dict[str, Any], data: Dict[str, Any], paths: Dict[str, pathlib.Path]):
    """
    Creates and saves the summary table and plots, and adds image links to the report.
    Assumes Transects.plot_... methods save the plot if 'save_path' is provided.
    """
    
    # --- 1. Constants and Filenames ---
    MA_PLOT_FILENAME = f"transect_biomass_comparison_{metrics['Confidence Threshold']}.png"
    OVP_PLOT_FILENAME = f"observed_vs_predicted_biomass_{metrics['Confidence Threshold']}.png"
    REPORT_FILENAME = "transect_summary.md"
    
    # Define a clean ordered list of metrics for the table
    METRIC_ORDER = [
        ("Confidence Threshold", metrics["Confidence Threshold"], ""),
        (None, None, None), # Separator
        ("Total Labeled Box Weight", metrics["total_lbl_box_weight"], "g"),
        ("Total Labeled Polygon Weight", metrics["total_poly_weight"], "g"),
        ("Total Inferred Weight", metrics["total_infer_weight"], "g"),
        (None, None, None), # Separator
        ("Inferred True Positive Weight", metrics["true_fish_weight"], "g"),
        ("Inferred False Positive Weight", metrics["false_fish_weight"], "g"),
        ("Labeled Missed Weight (False Neg.)", metrics["missed_fish_weight"], "g"),
        (None, None, None), # Separator
        ("Average Inferred Biomass Density", metrics["avg_infer_biomass"], "g/m²"),
        ("Average Labeled Box Biomass Density", metrics["avg_lblbox_biomass"], "g/m²"),
        ("Average Labeled Polygon Biomass Density", metrics["avg_lblpoly_biomass"], "g/m²")
    ]
    
    # Create the DataFrame cleanly from the ordered list
    data_for_table = {
        'Metric': [m[0] if m[0] is not None else '---' for m in METRIC_ORDER],
        'Value': [m[1] for m in METRIC_ORDER],
        'Units': [m[2] if m[2] is not None else '' for m in METRIC_ORDER]
    }
    df = pd.DataFrame(data_for_table)

    # Formatting: Handle separators, NaNs, and numerical formatting
    def fmt_val(x):
        if pd.isna(x):
            return 'N/A'
        if isinstance(x, numbers.Number):
            return f"{x:,.2f}"
        return str(x) if x is not None else ''
    df['Value'] = df['Value'].apply(fmt_val)
    df = df.replace('---', '')
    
    # --- 2. Generate and Trigger Plot Saving ---
    
    # Re-calculate biomass DFs for plotting 
    infer_biomass_df = Transects.biomass_transects(data["infer_transects"])
    lblbox_biomass_df = Transects.biomass_transects(data["lblres_transects"])
    lblpoly_biomass_df = Transects.biomass_transects(data["poly_assmt_transects"])

    # Track plot saving status
    ma_plot_saved = False
    ovp_plot_saved = False

    # Save Moving Average Plot
    try:
        # Pass the path to trigger saving inside the Transects method
        Transects.plot_biomass_comparison_moving_average(
            primary_transect=infer_biomass_df,
            primary_lbl="Inferred Biomass",
            primary_window=100,
            secondary_transect=lblbox_biomass_df if not lblbox_biomass_df.empty else None, 
            secondary_lbl="Labeled Box Biomass" if not lblbox_biomass_df.empty else None,
            secondary_window=100,
            tertiary_transect=lblpoly_biomass_df if not lblpoly_biomass_df.empty else None, 
            tertiary_lbl="Labeled Polygon Biomass" if not lblpoly_biomass_df.empty else None,
            tertiary_window=100,
            save_path=paths["transect_folder"] / MA_PLOT_FILENAME # <-- Pass path to trigger save
        )       
        ma_plot_saved = True
        print(f"Moving average plot saved to: {MA_PLOT_FILENAME}")
    except Exception as e:
        print(f"Warning: Could not generate or save moving average plot. Error: {e}")

    # Save Observed vs Predicted Plot
    try:
        # Assuming your OVP function is Transects.plot_observed_vs_predicted_biomass 
        # (your provided code had Transects.create_observed_vs_predicted_plot)
        Transects.create_observed_vs_predicted_plot(
            # Often, for OVP, you compare inferred to the best label, e.g., poly
            lbl_df=lblpoly_biomass_df if not lblpoly_biomass_df.empty else None, 
            infer_df=infer_biomass_df, 
            save_path=paths["transect_folder"] / OVP_PLOT_FILENAME # <-- Pass path to trigger save
        )
        ovp_plot_saved = True
        print(f"Observed vs. Predicted plot saved to: {OVP_PLOT_FILENAME}")
    except Exception as e:
        print(f"Warning: Could not generate or save OVP plot. Error: {e}")


    # --- 3. Write/Append to Markdown File ---
    output_path_report = paths["transect_folder"] / REPORT_FILENAME
    
    # Extract collect_id dynamically for the header (robust)
    collect_id_val = metrics.get('Collect ID', 'Unknown')
    try:
        if not data['infer_transects'].empty:
            id_col_name = next((c for c in data['infer_transects'].columns if 'collect_id' in c.lower() or 'collectid' in c.lower()), None)
            if id_col_name is not None:
                collect_id_val = data['infer_transects'][id_col_name].iloc[0]
    except Exception:
        # fallback to metrics value already set
        pass

    with open(output_path_report, 'a') as f:
        f.write(f"\n# Transect Summary Report: {paths['transect_folder'].name}\n\n")
        f.write(f"**Transect Run:** `{paths['run_folder'].name}` | **Collect ID:** `{collect_id_val}`\n\n")
        f.write("## 1. Biomass Metrics Summary\n\n")
        # to_markdown may require tabulate; provide a safe fallback
        try:
            f.write(df.to_markdown(index=False))
        except Exception:
            f.write(df.to_string(index=False))

    print(f"Summary report (including plot links) saved to: {output_path_report}")

# ======================================================================
# 5. MAIN EXECUTION
# ======================================================================

    
def main():

    args = arg_parser()
    
    print("--- Starting Report Generation Pipeline ---")
    print(f"Test Run: {args.test_run}, Transect: {args.collect_id}")
    
    try:
        # 1. Configuration
        confidence_threshold = args.confidence_threshold
        collect_id = args.collect_id
        test_run = args.test_run
        run_directory = args.run_directory
        paths = load_config(run_directory, test_run, collect_id, confidence_threshold)
        
        # 2. Run results for the confidence threshold (no inference)
        if args.run_results:
            command = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "runResults.py"),
                "--op_table", str(paths["op_table"]),
                "--metadata", str(paths["metadata"]),
                "--output_name", test_run,
                "--has_labels",
                "--results_confidence", str(confidence_threshold)
            ]
            # Use list form without shell=True for safety
            subprocess.run(command, check=True)
        # 3. Data Loading and Filtering
        data = load_and_filter_data(paths, collect_id, confidence_threshold)
        # 4. Metric Calculation
        metrics = calculate_metrics(data)

        # 5. Report Generation
        generate_report(metrics, data, paths)
        print("\n--- Pipeline Complete ---")
        
        

    except Exception as e:
        print(f"\nFATAL ERROR during pipeline execution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

