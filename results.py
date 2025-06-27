from image_area import ImageArea
import pandas as pd
import numpy as np
from utils import *

class YOLOResults:
    def __init__(self, meta_path, yolo_infer_path, substrate_path, op_path, conf_thresh=0.001):
        self.meta_path = meta_path
        self.yolo_infer_path = yolo_infer_path
        self.substrate_path = substrate_path
        self.op_path = op_path
        self.conf_thresh = conf_thresh

    def combine_meta_pred_substrate(self, find_closest=False):
        """        Combines metadata, YOLO inference results, and substrate predictions into a single DataFrame.
        Args:
            find_closest (bool): If True, will attempt to find the closest metadata match for each inference result.
        Returns:
            pd.DataFrame: A DataFrame containing combined metadata, YOLO inference results, and substrate predictions.
        """
        # Load required paths
        op_path = self.op_path
        yolo_infer_path = self.yolo_infer_path
        substrate_path = self.substrate_path
        meta_path = self.meta_path


        # Load YOLO inference results
        infer = None
        try:
            infer = pd.read_csv(yolo_infer_path, index_col=0)
            infer["conf"] = infer["conf"].astype(float)
            infer = infer[infer["conf"] >= 0.1]
            infer["Filename"] = infer["Filename"].str.replace("CI_", "PI_").apply(lambda x: x.split(".")[0])
            print("Total objects in inference", infer.shape)
            n_im_inferred = infer["Filename"].nunique()
            print("Number of images inferred", n_im_inferred)
        except Exception as e:
            print("Error loading YOLO inference results:", e)
            return None

        # Load substrate predictions
        substrate = None
        if substrate_path is not None and isinstance(substrate_path, str) and substrate_path.strip() != "":
            try:
                substrate = pd.read_csv(substrate_path, index_col=0)
                substrate = substrate.drop_duplicates(subset="Filename")
                print("Images with substrate predicted", substrate.shape)
            except Exception:
                print("Unable to read substrate csv from the path provided")
                substrate = None
        if substrate is None:
            # initialized df_combined as just the inference
            df_combined = infer
            print("No substrate inference provided")
        else:
            # Merge with substrate predictions and initialize combined dataframe
            df_combined = pd.merge(infer, substrate, how="left", on="Filename")
            assert len(df_combined) == len(infer)
            
        # Only check for mismatch if substrate is not a dummy DataFrame
        if substrate_path is not None and isinstance(substrate_path, str) and substrate_path.strip() != "":
            if len(df_combined) != len(infer):
                raise ValueError("Mismatch in lengths after merging inference and substrate data.")

        # Load metadata
        metadata = None
        if meta_path is not None and isinstance(meta_path, str) and meta_path.strip() != "":
            print("Loading metadata from", meta_path)
            try:
                metadata = pd.read_csv(meta_path, low_memory=False)
                filename_col = metadata.columns[metadata.columns.str.lower() == "filename"]
                if not filename_col.empty:
                    # clean Filename column
                    metadata["Filename"] = metadata[filename_col[0]].apply(lambda x: x.split(".")[0])
            except Exception:
                print("No metadata csv provided, returning inference results without metadata")
                metadata = None
        if metadata is None:
            print("No metadata provided, returning inference results without metadata")  # <-- add this print
        else:
            print("Shape of metadata csv", metadata.shape)
            # Merge metadata and inference results
            if not find_closest:
                # Direct merge on Filename
                df_combined = pd.merge(metadata, infer, how="left", on="Filename").sort_values(by="Filename").reset_index(drop=True)
            else:
                # Find closest metadata match for each inference result
                df_combined = pd.merge_asof(metadata.sort_values("Filename"), infer.sort_values("Filename"), on="Filename", direction="nearest")
            if df_combined["CollectID"].isna().any():
                print("Some rows have missing CollectID after merging metadata and inference results.")  # <-- add this print
                # Optionally: raise ValueError("Some rows have missing CollectID after merging metadata and inference results.")
            print("Combined infer and metadata", df_combined.shape)
        
        # Merge with site IDs
        site_ids = None
        if op_path is not None and isinstance(op_path, str) and op_path.strip() != "":
            try:
                # Load site IDs
                site_ids = pd.read_excel(op_path)[["COLLECT_ID", "SURVEY123_NAME", "LAKE"]].rename(columns={"COLLECT_ID": "CollectID"})
            except Exception as e:
                print("Error loading site IDs from OP table:", e)
    
            if site_ids is not None:
                # Merge with site IDs on the CollectID field which is populated from the metadata merge
                if not "CollectID" in df_combined.columns:
                    print("No survey operations table information was merged due to missing CollectID field")
                else:
                    df_combined = pd.merge(df_combined, site_ids, on="CollectID", how="left")
                    print("After merging with survey operations table", df_combined.shape)

        # Consistency check if metadata was provided
        if meta_path is not None and isinstance(meta_path, str) and meta_path.strip() != "":
            expected_rows = metadata.shape[0] - n_im_inferred + infer.shape[0]
            if expected_rows != df_combined.shape[0]:
                print(f"Warning, some discrepancy when joining metadata, substrate, and inferred data: {metadata.shape[0]} - {n_im_inferred} + {infer.shape[0]} != {df_combined.shape[0]}")
        
        return df_combined

    def clean_yolo_results(self, **kwargs):
        conf_thresh = self.conf_thresh
        df_combined = self.combine_meta_pred_substrate(**kwargs)
        columns_ = [
            "Time_s", "Filename", "SURVEY123_NAME", "LAKE", "Fish_ID", "x", "y", "w", "h", "conf", "conf_pass",
            "imw", "imh", "detect_id", "ground_truth_id", "Latitude", "Longitude",
            "DepthFromSurface_m", "DistanceToBottom_m", "Speed_kn", "Time_UTC", "image_path",
            "CollectID", "PS_mm", "ImageArea_m2", "year", "month", "day", "time",
            "box_DL_px", "box_DL_Cor_px", "box_DL_mm", "box_DL_mm_corr",
            "box_DL_weight_g", "box_DL_weight_g_corr", "substrate_class_2c"
        ]
        # Create an empty DataFrame with the desired columns
        yolores = pd.DataFrame(columns=columns_)
        df = df_combined.loc[:, df_combined.columns.isin(columns_)]
        join_columns = yolores.columns[yolores.columns.isin(df.columns)]
        # align columns
        yolores[join_columns] = df[join_columns]

        yolores["conf"] = yolores.conf.fillna(0)
        yolores["conf_pass"] = np.where(yolores.conf < conf_thresh, 0, 1)

        if yolores[["year", "month", "day", "time"]].isna().all().all():
            print("No date information in metadata, adding dates from timestamp")
            Time_s = yolores.Time_s
            yolores["year"] = Time_s.apply(lambda x: ReturnTime.get_Y(x))
            yolores["month"] = Time_s.apply(lambda x: ReturnTime.get_m(x))
            yolores["day"] = Time_s.apply(lambda x: ReturnTime.get_d(x))
            yolores["time"] = Time_s.apply(lambda x: ReturnTime.get_t(x))

        # Remove duplicate index values before assigning Fish_ID
        yolores = yolores.reset_index(drop=True)
        yolores = yolores.drop_duplicates()  # <-- Add this line to ensure no duplicate rows

        # Calculate fish detections and per-image stats
        fdac = yolores.conf_pass.sum()
        nmac = yolores.Filename.nunique() if yolores.Filename.notna().any() else 1
        print(f"fish detections @ conf >= {conf_thresh}", fdac)
        print(f"fish det per image@ conf >= {conf_thresh}", f"{fdac/nmac:0.2f}")

        # Assign Fish_ID only to passed detections, then fill for all
        yolores["Fish_ID"] = (
            yolores[yolores.conf_pass == 1]
            .groupby("Filename")
            .cumcount() + 1
        )
        # yolores["Fish_ID"] = yolores["Fish_ID"].fillna()
        # yolores["detect_id"] = yolores["detect_id"].fillna()
        yolores = yolores.reset_index(drop=True)
        return yolores

    def indices(self, yolores):
        collect_id = yolores['CollectID']
        CAM = collect_id.apply(lambda cid: cid.split('_')[-1])
        DRONE = collect_id.apply(lambda cid: cid.split('_')[-2])
        CD = collect_id.apply(lambda cid: int(cid.split('_')[0]))
        NF_idx = yolores[yolores.x.isna() & yolores.y.isna() & yolores.w.isna() & yolores.h.isna()].index
        ABS1_idx = collect_id[CAM == "ABS1"].index
        ABS2_idx = collect_id[CAM == "ABS2"].index
        Iver3069_idx = collect_id[DRONE == "Iver3069"].index
        Iver3098_idx = collect_id[DRONE == "Iver3098"].index
        REMUS03243_idx = collect_id[DRONE == "REMUS03243"].index
        ABS2_3069_12mm_idx = ABS2_idx.intersection(Iver3069_idx).intersection(CD[CD >= 20220716].index).intersection(CD[CD < 20220822].index)
        ABS2_3069_16mm_idx = ABS2_idx.intersection(Iver3069_idx).intersection(CD[CD >= 20220822].index)
        ABS2_3098_12mm_idx = ABS2_idx.intersection(Iver3098_idx).intersection(CD[CD >= 20220706].index).intersection(CD[CD < 20220819].index)
        ABS2_3098_16mm_idx = ABS2_idx.intersection(Iver3098_idx).intersection(CD[CD >= 20220819].index)
        assert len(ABS1_idx) + len(ABS2_3069_12mm_idx) + len(ABS2_3069_16mm_idx) + len(ABS2_3098_12mm_idx) + len(ABS2_3098_16mm_idx) + len(REMUS03243_idx) == len(yolores)
        assert len(ABS1_idx) + len(ABS2_idx) + len(REMUS03243_idx) == len(yolores)
        return NF_idx, ABS1_idx, ABS2_idx, ABS2_3069_12mm_idx, ABS2_3069_16mm_idx, ABS2_3098_12mm_idx, ABS2_3098_16mm_idx, REMUS03243_idx

    def area_and_pixel_size(self, yolores):
        yolores = yolores.copy()
        l = yolores.shape[0]
        h_ = np.zeros(l)
        f = np.zeros(l)
        N = np.zeros(l)
        W = yolores['DistanceToBottom_m'] * 1000
        NF_idx, ABS1_idx, ABS2_idx, ABS2_3069_12mm_idx, ABS2_3069_16mm_idx, ABS2_3098_12mm_idx, ABS2_3098_16mm_idx, REMUS03243_idx = self.indices(yolores)
        N[ABS1_idx] = N[ABS2_idx] = N[REMUS03243_idx] = 4112
        h_[ABS1_idx], h_[ABS2_idx], h_[REMUS03243_idx] = 14.2, 14.1, 13.9
        x, y, w, h = yolores.x.copy(), yolores.y.copy(), yolores.w.copy(), yolores.h.copy()
        x[NF_idx], y[NF_idx], w[NF_idx], h[NF_idx] = 0, 0, 0, 0
        yolores.x, yolores.y, yolores.w, yolores.h = x, y, w, h
        imgh, imgw = yolores.imh.astype(int), yolores.imw.astype(int)
        f[ABS1_idx], f[ABS2_3069_12mm_idx], f[ABS2_3069_16mm_idx] = 16, 12, 16
        f[ABS2_3098_12mm_idx], f[ABS2_3098_16mm_idx], f[REMUS03243_idx] = 12, 16, 12
        f = f.astype(int)
        HFOV = ImageArea.HFOV_func(W, h_, f)
        PS = ImageArea.PS_func(HFOV, N)
        yolores.PS_mm = PS
        image_ratio = imgh / imgw
        VFOV = image_ratio * HFOV
        area = (HFOV * VFOV) / 1e6
        yolores.ImageArea_m2 = area
        return yolores

    def calc_fish_wt(self, yolores):
            # Calculate the pixel size of the fish box, multiplied by 1 or 0 set by the conf_threshold
            yolores['box_DL_px'] = ImageArea.DL_px_func(
                yolores['w'] * yolores['imw'] * yolores['conf_pass'],
                yolores['h'] * yolores['imh'] * yolores['conf_pass']
            )
            yolores['box_DL_px'] = yolores['box_DL_px'].fillna(0)
            # Convert YOLO diagonal length to polynomial length
            yolores['box_DL_Cor_px'] = ImageArea.correct_DL_px_func(
                yolores['box_DL_px'], yolores['conf_pass']
            )
            yolores['box_DL_mm'] = ImageArea.calc_DL_mm_func(
                yolores['box_DL_Cor_px'], yolores['PS_mm']
            )
            yolores['box_DL_mm'] = yolores['box_DL_mm'].fillna(0)
            # Estimate fish weight based on the linear regression for converting fish length to weight
            yolores['box_DL_weight_g'] = ImageArea.calc_weight_func(yolores['box_DL_mm'])
            return yolores

    def calc_fish_wt_corr(self, yolores):
            ''' The calculated scaling factor using the calibration images'''
            NF_idx, ABS1_idx, ABS2_idx, ABS2_3069_12mm_idx, ABS2_3069_16mm_idx, ABS2_3098_12mm_idx, ABS2_3098_16mm_idx, REMUS03243_idx = self.indices(yolores)
            SF = np.zeros(len(yolores))
            ''' Use the following scaling factor if the original images have not been undistorted '''
            SF[ABS1_idx] = 1.295 * 0.9
            SF[ABS2_3069_12mm_idx] = 1.018675609 * 0.9
            SF[ABS2_3069_16mm_idx] = 0.90795 * 0.9
            SF[ABS2_3098_12mm_idx] = 1.018675609 * 0.9
            SF[ABS2_3098_16mm_idx] = 0.90795 * 0.9
            SF[REMUS03243_idx] = 1  # <------------------------------------------------- PLACEHOLDER Needs to be calibrated
            assert len(SF[ABS1_idx]) + len(SF[ABS2_3069_12mm_idx]) + len(SF[ABS2_3069_16mm_idx]) + len(SF[ABS2_3098_12mm_idx]) + len(SF[ABS2_3098_16mm_idx]) + len(SF[REMUS03243_idx]) == len(yolores)

            yolores['box_DL_mm_corr'] = ImageArea.apply_scaling_func(
                yolores['box_DL_mm'], SF
            )
            yolores['box_DL_mm_corr'] = yolores['box_DL_mm_corr'].fillna(0)
            yolores['box_DL_weight_g_corr'] = ImageArea.calc_weight_func(yolores['box_DL_mm_corr'])
            return yolores

    def yolo_results(self, **kwargs):
        """
        Main method to process YOLO results.
        Combines metadata, predictions, and substrate info,
        cleans the results, calculates area and pixel size,
        and computes fish weights.
        """
        yolores = self.clean_yolo_results()
        if not yolores[["Latitude", "Longitude"]].isna().all().all():
            yolores = self.area_and_pixel_size(yolores)
            yolores = self.calc_fish_wt(yolores)
            yolores = self.calc_fish_wt_corr(yolores)
        return yolores

if __name__ == "__main__":
    # Example usage:
    meta_path = "path/to/metadata.csv"
    yolo_infer_path = "path/to/yolo_inference.csv"
    substrate_path = "path/to/substrate_predictions.csv"
    op_path = "path/to/OP_table.xlsx" # operations table at collect level with Suvey123 information
    conf_thresh = 0.001

    # Initialize the output class
    output = YOLOResults(meta_path, yolo_infer_path, substrate_path, op_path, conf_thresh)

    yolores = output.yolo_results()

    # Save or further process yolores as needed
    print(yolores.head())



