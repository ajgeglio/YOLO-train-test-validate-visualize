import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pathlib

# --- 1. SETUP & CONFIGURATION ---
SCRIPT_DIR = pathlib.Path(__file__).parent if '__file__' in locals() else pathlib.Path.cwd()
sys.path.append(str(SCRIPT_DIR.parent.parent / "src"))
try:
    from utils import Utils
except ImportError:
    print(f"WARNING: Could not import 'Utils' from {str(SCRIPT_DIR.parent.parent / "src")}. Ensure the path is correct.")

# Configuration Dictionary - EDIT THIS SECTION
CONFIG = {
    "PATHS": {
        "OP_TABLE": Path(r"Z:\__AdvancedTechnologyBackup\07_Database\OP_TABLE.xlsx"),
        "METADATA_COMBINED": Path(r"Z:\__AdvancedTechnologyBackup\07_Database\MetadataCombined\all_annotated_images_metadata_20260121.csv"),
        "ADDL_SPECIES_LOG": Path(r"z:\__AdvancedTechnologyBackup\07_Database\addl_species_log.xlsx"),
        
        # Output Directories
        "OUTPUT_DIR": Path(r"Z:\__Organized_Directories_InProgress\GobyFinderDatasets\AUV_datasets"),
        "LOCAL_OUTPUT_DIR": Path(r"D:\datasets\goby"),
        "TILED_IM_DIR": Path(r"D:\datasets\goby\tiled"),
        "FULL_IM_DIR": Path(r"D:\datasets\goby\full"),
        
        # YOLO Dataset Folders
        "YOLO_DATASET_FOLDER": Path(r"D:\ageglio-1\gobyfinder_yolov8\datasets\AUV_datasets\run13"),
        
        # HNM (Hard Negative Mining) Paths
        "HNM_INFERENCE_RESULTS": Path(r"Z:\__AdvancedTechnologyBackup\03_InferenceResults\GobyFinder\HNM data 2"),
    },
    
    # Collect IDs for splitting data
    "COLLECT_IDS": {
        "transects": {
            "20200806_001_Iver3069_ABS1", "20200816_001_Iver3069_ABS1", "20210825_001_Iver3069_ABS1", 
            "20210720_001_Iver3069_ABS1", "20240618_001_Iver3069_ABS2", "20240804_001_Iver3069_ABS2"
        },
        "test": {
            "20200809_001_Iver3069_ABS1", "20200818_001_Iver3069_ABS1", "20200902_001_Iver3069_ABS1", 
            "20200820_001_Iver3069_ABS1", "20200821_001_Iver3069_ABS1", "20200823_001_Iver3069_ABS1",
            "20210811_001_Iver3069_ABS1", "20210812_001_Iver3069_ABS1", "20210812_002_Iver3069_ABS1", 
            "20210719_001_Iver3069_ABS1", "20210829_001_Iver3069_ABS1", "20210911_001_Iver3069_ABS1", 
            "20210911_002_Iver3069_ABS1", "20210925_001_Iver3069_ABS1", "20220624_001_Iver3069_ABS1", 
            "20220714_002_Iver3069_ABS1", "20220727_001_Iver3069_ABS2", "20220811_002_Iver3098_ABS2", 
            "20220807_003_Iver3069_ABS2", "20220901_001_Iver3069_ABS2", "20220814_001_Iver3069_ABS2", 
            "20220814_002_Iver3069_ABS2", "20230710_001_Iver3098_ABS2", "20230909_001_Iver3069_ABS2", 
            "20230810_002_Iver3098_ABS2", "20230727_001_Iver3098_ABS2",
            "20250917_001_REMUS03243_VCC"
        },
        "validation": {
            "20200916_001_Iver3069_ABS1", "20200922_002_Iver3069_ABS1", "20200923_002_Iver3069_ABS1",
            "20210712_001_Iver3069_ABS1", "20210909_001_Iver3069_ABS1", "20210920_001_Iver3069_ABS1", 
            "20210707_001_Iver3069_ABS1", "20210912_001_Iver3069_ABS1", "20210912_002_Iver3069_ABS1", 
            "20210913_001_Iver3069_ABS1", "20220711_002_Iver3069_ABS1", "20220714_003_Iver3069_ABS1", 
            "20220717_001_Iver3098_ABS2", "20220825_001_Iver3098_ABS2", "20220914_002_Iver3069_ABS2", 
            "20220902_001_Iver3069_ABS2", "20230802_001_Iver3098_ABS2", "20230625_001_Iver3098_ABS2", 
            "20230718_002_Iver3098_ABS2", "20230811_001_Iver3098_ABS2", "20230715_001_Iver3098_ABS2",
            "20250909_001_REMUS03243_VCC"
        }
    }
}

# --- 2. HELPER FUNCTIONS ---
def choose_best_column(df, candidates):
    existing = [c for c in candidates if c in df.columns]
    if not existing:
        return None

    if len(existing) == 1:
        return existing[0]

    # Pick the one with fewer missing values
    return min(existing, key=lambda c: df[c].isna().sum())


def get_split_name(collect_id, collect_ids_dict):
    """Determines the split name for a given collect_id."""
    for split_name, ids in collect_ids_dict.items():
        if collect_id in ids:
            return split_name
    return 'train'

def process_metadata(config):
    """Loads, filters, and splits metadata."""
    print("Loading and preparing METADATA_COMBINED...")
    paths = config["PATHS"]
    
    # Load Data
    all_annotated_meta = pd.read_csv(paths["METADATA_COMBINED"], low_memory=False)
    addl_species_filenames = pd.read_excel(paths["ADDL_SPECIES_LOG"]).Filename.tolist()

    # --- Normalize CollectID column ---
    collect_candidates = ["CollectID", "collect_id"]
    best_collect_id = choose_best_column(all_annotated_meta, collect_candidates)

    if best_collect_id is None:
        raise ValueError("No CollectID-like column found in metadata")

    # Create unified CollectID column
    all_annotated_meta["CollectID"] = all_annotated_meta[best_collect_id]

    # Optionally drop the redundancy
    cols_to_drop = [c for c in collect_candidates if c in all_annotated_meta.columns and c != "CollectID"]
    all_annotated_meta.drop(columns=cols_to_drop, inplace=True)

    # Apply Split Labels
    all_annotated_meta['split'] = all_annotated_meta["CollectID"].apply(
        lambda x: get_split_name(x, config["COLLECT_IDS"])
    )

    # Filtering Logic
    # 1. Usable, Valid Year/Fish count logic, Not Null distance
    # 2. Exclude files in addl_species_log
    query_str = (
        'Usability == "Usable" and '
        '((year == 2021 and n_fish >= 2) or (year in [2020, 2022, 2023, 2024, 2025, 2026, 2027])) and '
        'DistanceToBottom_m.notnull()'
    )
    filtered_meta = all_annotated_meta.query(query_str)
    filtered_meta = filtered_meta[~filtered_meta['Filename'].isin(addl_species_filenames)]
    filtered_meta.to_csv(r"Z:\__AdvancedTechnologyBackup\07_Database\MetadataCombined\filtered_annotated_metadata.csv")
    print(f"Filtered Usable data shape: {filtered_meta.shape}")
    
    # Save Split Text Files
    paths["OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
    paths["LOCAL_OUTPUT_DIR"].mkdir(parents=True, exist_ok=True)
    
    for split_name in ['train', 'test', 'validation', 'transects']:
        filenames = filtered_meta[filtered_meta['split'] == split_name]['Filename'].tolist()
        
        # Write to local output dir
        out_file = paths["LOCAL_OUTPUT_DIR"] / f"{split_name}.txt"
        Utils.write_list_txt(filenames, out_file)
        print(f"Saved {len(filenames)} filenames to {out_file}")

    return filtered_meta

def generate_statistics(filtered_meta, config):
    """Generates summary statistics CSVs."""
    print("\n--- Generating statistics reports ---")
    paths = config["PATHS"]
    
    # Aggregation
    df_grouped = filtered_meta.groupby(["CollectID", "split"]).agg(
        n_images=('Filename', 'count'),
        n_fish_p_collect=('n_fish', 'sum')
    ).reset_index()

    df_grouped['n_fish_p_image'] = df_grouped['n_fish_p_collect'] / df_grouped['n_images']
    df_grouped['year'] = df_grouped['CollectID'].str[:4].astype(int)
    
    # Merge with OP_TABLE
    op_table = pd.read_excel(paths["OP_TABLE"])
    df_stats = df_grouped.merge(
        op_table[["COLLECT_ID", "LAKE_NAME", "MISSION_NAME", "PORT_NAME", "LATITUDE", "LONGITUDE"]],
        left_on="CollectID",
        right_on="COLLECT_ID",
        how='left'
    ).drop(columns='COLLECT_ID')

    # Save Stats
    df_stats.to_csv(paths["OUTPUT_DIR"] / "Run13_collect_stats.csv", index=False)
    df_stats.to_csv(paths["LOCAL_OUTPUT_DIR"] / "Run13_collect_stats.csv", index=False)
    
    # Print Summary
    for split_name, df_split in df_stats.groupby('split'):
        print(f"Total images for {split_name}: {df_split['n_images'].sum()}")
        
    return df_stats

def write_yolo_lists_full(split, config):
    """Generates YOLO txt lists for FULL images."""
    paths = config["PATHS"]
    full_dataset_folder = paths["YOLO_DATASET_FOLDER"] / "full" / split
    full_dataset_folder.mkdir(parents=True, exist_ok=True)
    
    # Load expected images from the split text file we created earlier
    split_txt_path = paths["LOCAL_OUTPUT_DIR"] / f"{split}.txt" # Changed to LOCAL_OUTPUT_DIR for consistency
    imgs = Utils.read_list_txt(str(split_txt_path))
    
    # Find actual files on disk
    search_path = paths["FULL_IM_DIR"] / split
    all_image_paths = list(search_path.glob("images/*.png")) + list(search_path.glob("images/*.jpg"))
    all_label_paths = list(search_path.glob("labels/*.txt"))
    
    # Convert Paths to strings for Utils compatibility if necessary
    all_image_paths_str = [str(p) for p in all_image_paths]
    all_label_paths_str = [str(p) for p in all_label_paths]

    images, labels = Utils.list_full_set(imgs, all_image_paths_str, all_label_paths_str)
    
    # Validation
    if len(imgs) != len(all_image_paths):
        print(f"[{split}] Warning: Mismatch between split list ({len(imgs)}) and local files ({len(all_image_paths)})")
        
    print(f"Number of full images for {split}: {len(images)}")
    
    # Write lists
    Utils.write_list_txt(images, str(full_dataset_folder / "images.txt"))
    Utils.write_list_txt(labels, str(full_dataset_folder / "labels.txt"))

def write_yolo_lists_tiled(split, config):
    """Generates YOLO txt lists for TILED images."""
    paths = config["PATHS"]
    tiled_dataset_folder = paths["YOLO_DATASET_FOLDER"] / "tiled" / split
    tiled_dataset_folder.mkdir(parents=True, exist_ok=True)
    
    tiled_image_folder = paths["TILED_IM_DIR"] / split / "images"
    tiled_label_folder = paths["TILED_IM_DIR"] / split / "labels"
    
    images = list(tiled_image_folder.glob("*.png")) + list(tiled_image_folder.glob("*.jpg"))
    labels = list(tiled_label_folder.glob("*.txt"))
    
    print(f"Tiled images for {split}: {len(images)}")
    
    # Convert to strings for Utils
    images_str = [str(p) for p in images]
    labels_str = [str(p) for p in labels]
    
    Utils.write_list_txt(images_str, str(tiled_dataset_folder / "images.txt"))
    Utils.write_list_txt(labels_str, str(tiled_dataset_folder / "labels.txt"))
    
    return len(images)

def convert_validation_to_jpg(config):
    """Converts validation PNGs to JPGs for Dataloader compatibility."""
    paths = config["PATHS"]
    val_image_dir = paths["TILED_IM_DIR"] / "validation" / "images"
    png_backup_dir = paths["TILED_IM_DIR"] / "validation" / "png" / "images"
    
    print("\nConverting Validation PNGs to JPG...")
    # Get all PNGs currently in the folder
    png_files = list(val_image_dir.glob("*.png"))
    
    if not png_files:
        print("No PNG files found in validation folder. Skipping conversion.")
        return

    # List to keep track of successfully converted files to move later
    files_to_move = []

    for png_path in tqdm(png_files):
        jpg_path = png_path.with_suffix('.jpg')
        
        # --- SAFETY CHECK ---
        # If the JPG already exists, skip the conversion to save time
        if jpg_path.exists():
            files_to_move.append(str(png_path))
            continue 
        # --------------------

        try:
            Utils.convert_png_to_highest_quality_jpeg(str(png_path), str(jpg_path))
            files_to_move.append(str(png_path))
        except Exception as e:
            print(f"Failed to convert {png_path.name}: {e}")
    
    # Move PNGs if they were successfully processed or already existed
    if files_to_move:
        print(f"Moving {len(files_to_move)} processed PNGs to backup...")
        png_backup_dir.mkdir(parents=True, exist_ok=True)
        Utils.MOVE_files_lst(files_to_move, str(png_backup_dir))

def perform_hard_negative_mining(config):
    """
    Filters the training set to remove 'easy' negatives (backgrounds) 
    that don't contribute to learning, preventing exploding gradients.
    """
    print("\n--- Starting Hard Negative Mining (HNM) ---")
    paths = config["PATHS"]
    
    lbl_report_path = paths["HNM_INFERENCE_RESULTS"] / "label_box_results.csv"
    scores_path = paths["HNM_INFERENCE_RESULTS"] / "scores.csv"
    
    if not lbl_report_path.exists() or not scores_path.exists():
        print("HNM files not found. Skipping HNM step.")
        return

    lbl_report = pd.read_csv(lbl_report_path, low_memory=False)
    scores = pd.read_csv(scores_path, index_col=0)

    # 1. Identification
    positive_filenames = lbl_report[~lbl_report.conf.isna()].Filename.unique()
    
    # 2. Hardest Negatives (Background tiles with confidence >= 0.2)
    # Note: Conf >= 0.2 on a background tile implies the model strongly thinks it sees something (False Positive)
    hardest_negatives = scores[(scores.ground_truth_id.isna()) & (scores.conf >= 0.2)]
    hardest_negative_filenames = hardest_negatives.Filename.unique()

    # 3. Combine Lists
    all_filenames_to_keep = list(positive_filenames) + list(hardest_negative_filenames)
    print(f"Total tiles to keep in training (HNM): {len(all_filenames_to_keep)}")

    # 4. Write Filtered Training Lists
    split = "train"
    tile_img_dir = paths["TILED_IM_DIR"] / split / "images"
    tile_lbl_dir = paths["TILED_IM_DIR"] / split / "labels"
    dataset_folder = paths["YOLO_DATASET_FOLDER"] / "tiled" / split
    
    # Construct paths using pathlib
    all_tiles_to_keep = [str(tile_img_dir / (f + ".png")) for f in all_filenames_to_keep]
    all_labels_to_keep = [str(tile_lbl_dir / (f + ".txt")) for f in all_filenames_to_keep]
    
    assert len(all_tiles_to_keep) == len(all_labels_to_keep)
    
    # Write
    dataset_folder.mkdir(parents=True, exist_ok=True)
    Utils.write_list_txt(all_tiles_to_keep, str(dataset_folder / "images.txt"))
    Utils.write_list_txt(all_labels_to_keep, str(dataset_folder / "labels.txt"))
    
    # Sanity Check
    sanity_check = Utils.read_list_txt(str(dataset_folder / "images.txt"))
    print(f"HNM Complete. Final training set size: {len(sanity_check)}")

# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Filter Metadata
    filtered_df = process_metadata(CONFIG)
    
    # 2. Statistics
    stats_df = generate_statistics(filtered_df, CONFIG)
    
    # 3. Full Image Lists
    for split in ["train", "test", "validation"]:
        write_yolo_lists_full(split, CONFIG)
        
    # 4. Convert Validation to JPG (Optional but recommended by user)
    # Note: Only run this if your validation set is currently PNGs in the tiled dir
    convert_validation_to_jpg(CONFIG)

    # 5. Tiled Image Lists (Initial pass)
    total_images = 0
    for split in ["train", "test", "validation"]:
        n = write_yolo_lists_tiled(split, CONFIG)
        total_images += n
        
    # 6. Hard Negative Mining (Overwrite Training List)
    perform_hard_negative_mining(CONFIG)