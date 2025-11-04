from image_area import ImageArea
import pandas as pd
import numpy as np
from utils import *

class YOLOResults:
    def __init__(self, meta_path, yolo_infer_path, substrate_path, op_path, conf_thresh=0.1):
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
            infer = infer.rename(columns={"imw": "imw_p", "imh": "imh_p"})
            infer["conf"] = infer["conf"].astype(float)
            infer = infer[infer["conf"] >= 0.099] # Self-imposed threshold for confidence to shorten results output
            print("Total objects in inference", infer.shape)
            n_im_inferred = infer["Filename"].nunique()
            im_inferred = infer["Filename"].unique()
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
                im_in_meta = metadata["Filename"].unique()
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
                site_ids = pd.read_excel(op_path)[["COLLECT_ID", "MISSION_NAME", "LAKE_NAME"]].rename(columns={"COLLECT_ID": "CollectID"})
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
                print("usually this is due to missing metadata for some images, or images that were not inferred")
                print(f"There are {len(list(set(im_inferred).difference(set(im_in_meta))))} images that were inferred but not in metadata")
                print(f"There are {len(list(set(im_in_meta).difference(set(im_inferred))))} images in metadata but not inferred")
                base_path = os.path.dirname(yolo_infer_path)
                with open(os.path.join(base_path, "discrepancy.txt"), "w") as f:
                    discrepant_images = list(set(im_inferred).difference(set(im_in_meta)))
                    f.write('\n'.join(discrepant_images))
            try:
                if df_combined.imw.all() != df_combined.imw_p.all():
                    print("Warning: Image width metadata does not match inferred image width.")
                if df_combined.imh.all() != df_combined.imh_p.all():
                    print("Warning: Image height metadata does not match inferred image height.")
            except Exception as e:
                print("Unable to check imw/imh metadata and inference consistency:", e)

        return df_combined

    def clean_yolo_results(self, **kwargs):
        conf_thresh = self.conf_thresh
        df_combined = self.combine_meta_pred_substrate(**kwargs)
        columns_ = [
            "Time_s", "Filename", "MISSION_NAME", "LAKE_NAME", "Fish_ID","cls", "x", "y", "w", "h", "conf", "conf_pass",
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
    
    def result_filt_g(df):
        return df[((df.box_DL_weight_g_corr<0.1) | (df.box_DL_weight_g_corr>80)) & (df.conf>0)]
    
    def filter_based_on_weight(self, yolores, yolo_infer_path):
        filt_predictions = YOLOResults.result_filt_g(yolores)
        yolores_filtered = yolores.drop(filt_predictions.index)
        filt_detect_ids = filt_predictions.detect_id
        pred_df = pd.read_csv(yolo_infer_path, index_col=0)
        n_to_remove_in_pred_df = len(list(set(pred_df.detect_id).intersection(filt_detect_ids)))
        print("filtering", n_to_remove_in_pred_df, "as outlier objects")
        pred_df_edit = pred_df[~pred_df.detect_id.isin(filt_detect_ids)]
        assert len(yolores) - len(filt_predictions) == len(yolores_filtered), "Results romoval count not correct"
        assert len(pred_df) - n_to_remove_in_pred_df == len(pred_df_edit), "Prediction romoval count not correct"
        pred_df_edit.to_csv(os.path.join(os.path.dirname(yolo_infer_path), "predictions_filtered.csv"))
        return yolores_filtered

    def yolo_results(self, **kwargs):
        """
        Main method to process YOLO results.
        Combines metadata, predictions, and substrate info,
        cleans the results, calculates area and pixel size,
        and computes fish weights.
        """
        yolores = self.clean_yolo_results()
        if not yolores["DistanceToBottom_m"].isna().all():
            yolores = ResultsUtils.area_and_pixel_size(yolores)
            yolores = ResultsUtils.calc_fish_wt(yolores)
            yolores = ResultsUtils.calc_fish_wt_corr(yolores)
            yolores = self.filter_based_on_weight(yolores, self.yolo_infer_path)
        return yolores

# if __name__ == "__main__":
#     # Example usage:
#     meta_path = "path/to/metadata.csv"
#     yolo_infer_path = "path/to/yolo_inference.csv"
#     substrate_path = "path/to/substrate_predictions.csv"
#     op_path = "path/to/OP_table.xlsx" # operations table at collect level with Suvey123 information
#     conf_thresh = 0.001

#     # Initialize the output class
#     output = YOLOResults(meta_path, yolo_infer_path, substrate_path, op_path, conf_thresh)

#     yolores = output.yolo_results()

#     # Save or further process yolores as needed
#     print(yolores.head())

class LBLResults:
    def __init__(self, meta_path, yolo_lbl_path, substrate_path, op_path):
        self.meta_path = meta_path
        self.yolo_lbl_path = yolo_lbl_path
        self.substrate_path = substrate_path
        self.op_path = op_path

    def combine_meta_lbl_substrate(self, find_closest=False):
        """        Combines metadata, YOLO label results, and substrate predictions into a single DataFrame.
        Args:
            find_closest (bool): If True, will attempt to find the closest metadata match for each label result.
        Returns:
            pd.DataFrame: A DataFrame containing combined metadata, YOLO label results, and substrate predictions.
        """
        # Load required paths
        op_path = self.op_path
        yolo_lbl_path = self.yolo_lbl_path
        substrate_path = self.substrate_path
        meta_path = self.meta_path


        # Load YOLO label results
        lbls = None
        try:
            lbls = pd.read_csv(yolo_lbl_path, index_col=0)
            lbls['conf'] = 1
            print("Total objects in labels", lbls.shape)
            n_im_lbld = lbls["Filename"].nunique()
            im_lbld = lbls["Filename"].unique()
            print("Number of images labeled", n_im_lbld)
        except Exception as e:
            print("Error loading YOLO label results:", e)
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
            df_combined = lbls
            print("No substrate inference provided")
        else:
            # Merge with substrate predictions and initialize combined dataframe
            df_combined = pd.merge(lbls, substrate, how="left", on="Filename")
            assert len(df_combined) == len(lbls)
            
        # Only check for mismatch if substrate is not a dummy DataFrame
        if substrate_path is not None and isinstance(substrate_path, str) and substrate_path.strip() != "":
            if len(df_combined) != len(lbls):
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
                im_in_meta = metadata["Filename"].unique()
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
                df_combined = pd.merge(metadata, lbls, how="left", on="Filename").sort_values(by="Filename").reset_index(drop=True)
            else:
                # Find closest metadata match for each inference result
                df_combined = pd.merge_asof(metadata.sort_values("Filename"), lbls.sort_values("Filename"), on="Filename", direction="nearest")
            if df_combined["CollectID"].isna().any():
                print("Some rows have missing CollectID after merging metadata and inference results.")  # <-- add this print
                # Optionally: raise ValueError("Some rows have missing CollectID after merging metadata and inference results.")
            print("Combined labels and metadata", df_combined.shape)
        
        # Merge with site IDs
        site_ids = None
        if op_path is not None and isinstance(op_path, str) and op_path.strip() != "":
            try:
                # Load site IDs
                site_ids = pd.read_excel(op_path)[["COLLECT_ID", "MISSION_NAME", "LAKE_NAME"]].rename(columns={"COLLECT_ID": "CollectID"})
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
            expected_rows = metadata.shape[0] - n_im_lbld + lbls.shape[0]
            if expected_rows != df_combined.shape[0]:
                print(f"Warning, some discrepancy when joining metadata, substrate, and labeled data: {metadata.shape[0]} - {n_im_lbld} + {lbls.shape[0]} != {df_combined.shape[0]}")
                print("usually this is due to missing metadata for some images, or images that were not labeled")
                print(f"There are {len(list(set(im_lbld).difference(set(im_in_meta))))} images that were labeled but not in metadata")
                print(f"There are {len(list(set(im_in_meta).difference(set(im_lbld))))} images in metadata but not labeled")
                base_path = os.path.dirname(yolo_lbl_path)
                with open(os.path.join(base_path, "discrepancy.txt"), "w") as f:
                    discrepant_images = list(set(im_lbld).difference(set(im_in_meta)))
                    f.write('\n'.join(discrepant_images))
            if df_combined.imw.all() != df_combined.imw_l.all():
                print("Warning: Image width metadata does not match inferred image width.")
            if df_combined.imh.all() != df_combined.imh_l.all():
                print("Warning: Image height metadata does not match inferred image height.")
                    
        return df_combined

    def clean_lbl_results(self, **kwargs):
        df_combined = self.combine_meta_lbl_substrate(**kwargs)
        columns_ = [
            "Time_s", "Filename", "MISSION_NAME", "LAKE_NAME", "Fish_lbl_ID", "cls", "x", "y", "w", "h", "conf", "conf_pass",
            "imw", "imh", "detect_id", "ground_truth_id", "Latitude", "Longitude",
            "DepthFromSurface_m", "DistanceToBottom_m", "Speed_kn", "Time_UTC", "image_path",
            "CollectID", "PS_mm", "ImageArea_m2", "year", "month", "day", "time",
            "box_DL_px", "box_DL_Cor_px", "box_DL_mm", "box_DL_mm_corr",
            "box_DL_weight_g", "box_DL_weight_g_corr", "substrate_class_2c"
        ]
        # Create an empty DataFrame with the desired columns
        lblres = pd.DataFrame(columns=columns_)
        df = df_combined.loc[:, df_combined.columns.isin(columns_)]
        join_columns = lblres.columns[lblres.columns.isin(df.columns)]
        # align columns
        lblres[join_columns] = df[join_columns]
        # lblres["conf"] = lblres.conf.fillna(1)
        lblres["conf_pass"] = 1

        if lblres[["year", "month", "day", "time"]].isna().all().all():
            print("No date information in metadata, adding dates from timestamp")
            Time_s = lblres.Time_s
            lblres["year"] = Time_s.apply(lambda x: ReturnTime.get_Y(x))
            lblres["month"] = Time_s.apply(lambda x: ReturnTime.get_m(x))
            lblres["day"] = Time_s.apply(lambda x: ReturnTime.get_d(x))
            lblres["time"] = Time_s.apply(lambda x: ReturnTime.get_t(x))

        # Remove duplicate index values before assigning Fish_ID
        lblres = lblres.reset_index(drop=True)
        lblres = lblres.drop_duplicates()  # <-- Add this line to ensure no duplicate rows

        # Calculate fish detections and per-image stats
        fdac = lblres.conf_pass.sum()
        nmac = lblres.Filename.nunique() if lblres.Filename.notna().any() else 1
        print(f"Total fish labels", fdac)
        print(f"Avr fish labels per image: {fdac/nmac:0.2f}")

        # Assign Fish_ID only to passed detections, then fill for all
        lblres["Fish_lbl_ID"] = (
            lblres[lblres.conf_pass == 1]
            .groupby("Filename")
            .cumcount() + 1
        )
        # lblres["Fish_ID"] = lblres["Fish_ID"].fillna()
        # lblres["ground_truth_id"] = lblres["ground_truth_id"].fillna()
        lblres = lblres.reset_index(drop=True)
        return lblres

    def lbl_results(self, **kwargs):
        """
        Main method to process YOLO results.
        Combines metadata, predictions, and substrate info,
        cleans the results, calculates area and pixel size,
        and computes fish weights.
        """
        lblres = self.clean_lbl_results()
        if not lblres["DistanceToBottom_m"].isna().all():
            lblres = ResultsUtils.area_and_pixel_size(lblres)
            lblres = ResultsUtils.calc_fish_wt(lblres)
            lblres = ResultsUtils.calc_fish_wt_corr(lblres)
        return lblres

# if __name__ == "__main__":
#     # Example usage:
#     meta_path = "path/to/metadata.csv"
#     yolo_lbl_path = "path/to/labels.csv"
#     substrate_path = "path/to/substrate_predictions.csv"
#     op_path = "path/to/OP_table.xlsx" # operations table at collect level with Suvey123 information

#     # Initialize the output class
#     output = LBLResults(meta_path, yolo_lbl_path, substrate_path, op_path)

#     lblres = output.lbl_results()

#     # Save or further process lblres as needed
#     print(lblres.head())

class ResultsUtils:
    """Class to calculate image area and pixel size based on YOLO results."""
    @staticmethod
    def indices(results):
        collect_id = results['CollectID']
        CAM = collect_id.apply(lambda cid: cid.split('_')[-1])
        DRONE = collect_id.apply(lambda cid: cid.split('_')[-2])
        CD = collect_id.apply(lambda cid: int(cid.split('_')[0]))
        NF_idx = results[results.x.isna() & results.y.isna() & results.w.isna() & results.h.isna()].index
        ABS1_idx = collect_id[CAM == "ABS1"].index
        ABS2_idx = collect_id[CAM == "ABS2"].index
        Iver3069_idx = collect_id[DRONE == "Iver3069"].index
        Iver3098_idx = collect_id[DRONE == "Iver3098"].index
        REMUS03243_idx = collect_id[DRONE == "REMUS03243"].index
        ABS2_3069_12mm_idx = ABS2_idx.intersection(Iver3069_idx).intersection(CD[CD >= 20220716].index).intersection(CD[CD < 20220822].index)
        ABS2_3069_16mm_idx = ABS2_idx.intersection(Iver3069_idx).intersection(CD[CD >= 20220822].index)
        ABS2_3098_12mm_idx = ABS2_idx.intersection(Iver3098_idx).intersection(CD[CD >= 20220706].index).intersection(CD[CD < 20220819].index)
        ABS2_3098_16mm_idx = ABS2_idx.intersection(Iver3098_idx).intersection(CD[CD >= 20220819].index)
        assert len(ABS1_idx) + len(ABS2_3069_12mm_idx) + len(ABS2_3069_16mm_idx) + len(ABS2_3098_12mm_idx) + len(ABS2_3098_16mm_idx) + len(REMUS03243_idx) == len(results)
        assert len(ABS1_idx) + len(ABS2_idx) + len(REMUS03243_idx) == len(results)
        return NF_idx, ABS1_idx, ABS2_idx, ABS2_3069_12mm_idx, ABS2_3069_16mm_idx, ABS2_3098_12mm_idx, ABS2_3098_16mm_idx, REMUS03243_idx

    @staticmethod
    def area_and_pixel_size(results):
        results = results.copy()
        l = results.shape[0]
        h_ = np.zeros(l)
        f = np.zeros(l)
        N = np.zeros(l)
        W = results['DistanceToBottom_m'] * 1000
        NF_idx, ABS1_idx, ABS2_idx, ABS2_3069_12mm_idx, ABS2_3069_16mm_idx, ABS2_3098_12mm_idx, ABS2_3098_16mm_idx, REMUS03243_idx = ResultsUtils.indices(results)
        N[ABS1_idx] = N[ABS2_idx] = N[REMUS03243_idx] = 4112
        h_[ABS1_idx], h_[ABS2_idx], h_[REMUS03243_idx] = 14.2, 14.1, 13.9
        x, y, w, h = results.x.copy(), results.y.copy(), results.w.copy(), results.h.copy()
        x[NF_idx], y[NF_idx], w[NF_idx], h[NF_idx] = 0, 0, 0, 0
        results.x, results.y, results.w, results.h = x, y, w, h
        imgh, imgw = results.imh.astype(int), results.imw.astype(int)
        f[ABS1_idx], f[ABS2_3069_12mm_idx], f[ABS2_3069_16mm_idx] = 16, 12, 16
        f[ABS2_3098_12mm_idx], f[ABS2_3098_16mm_idx], f[REMUS03243_idx] = 12, 16, 12
        f = f.astype(int)
        HFOV = ImageArea.HFOV_func(W, h_, f)
        PS = ImageArea.PS_func(HFOV, N)
        results.PS_mm = PS
        image_ratio = imgh / imgw
        VFOV = image_ratio * HFOV
        area = (HFOV * VFOV) / 1e6
        results.ImageArea_m2 = area
        return results

    @staticmethod
    def calc_fish_wt(results):
            # Calculate the pixel size of the fish box, multiplied by 1 or 0 set by the conf_threshold
            results['box_DL_px'] = ImageArea.DL_px_func(
                results['w'] * results['imw'] * results['conf_pass'],
                results['h'] * results['imh'] * results['conf_pass']
            )
            results['box_DL_px'] = results['box_DL_px'].fillna(0)
            # Convert YOLO diagonal length to polynomial length
            results['box_DL_Cor_px'] = ImageArea.correct_DL_px_func(
                results['box_DL_px'], results['conf_pass']
            )
            results['box_DL_mm'] = ImageArea.calc_DL_mm_func(
                results['box_DL_Cor_px'], results['PS_mm']
            )
            results['box_DL_mm'] = results['box_DL_mm'].fillna(0)
            # Estimate fish weight based on the linear regression for converting fish length to weight
            results['box_DL_weight_g'] = ImageArea.calc_weight_func(results['box_DL_mm'])
            return results

    @staticmethod
    def calc_fish_wt_corr(results):
            ''' The calculated scaling factor using the calibration images'''
            NF_idx, ABS1_idx, ABS2_idx, ABS2_3069_12mm_idx, ABS2_3069_16mm_idx, ABS2_3098_12mm_idx, ABS2_3098_16mm_idx, REMUS03243_idx = ResultsUtils.indices(results)
            SF = np.zeros(len(results))
            ''' Use the following scaling factor if the original images have not been undistorted '''
            SF[ABS1_idx] = 1.295 * 0.9
            SF[ABS2_3069_12mm_idx] = 1.018675609 * 0.9
            SF[ABS2_3069_16mm_idx] = 0.90795 * 0.9
            SF[ABS2_3098_12mm_idx] = 1.018675609 * 0.9
            SF[ABS2_3098_16mm_idx] = 0.90795 * 0.9
            SF[REMUS03243_idx] = 1  # <------------------------------------------------- PLACEHOLDER Needs to be calibrated
            assert len(SF[ABS1_idx]) + len(SF[ABS2_3069_12mm_idx]) + len(SF[ABS2_3069_16mm_idx]) + len(SF[ABS2_3098_12mm_idx]) + len(SF[ABS2_3098_16mm_idx]) + len(SF[REMUS03243_idx]) == len(results)

            results['box_DL_mm_corr'] = ImageArea.apply_scaling_func(
                results['box_DL_mm'], SF
            )
            results['box_DL_mm_corr'] = results['box_DL_mm_corr'].fillna(0)
            results['box_DL_weight_g_corr'] = ImageArea.calc_weight_func(results['box_DL_mm_corr'])
            return results