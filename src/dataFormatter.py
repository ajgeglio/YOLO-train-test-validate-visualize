import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Dict, Any

class YOLODataFormatter:
    """
    Processes raw YOLO prediction results and corresponding ground truth labels 
    into structured Pandas DataFrames.
    
    This class is specifically designed for small-scale testing scenarios, allowing 
    for easy data analysis, CSV saving, and advanced plotting of predictions (e.g., TP/FP visualization).
    """

    @staticmethod
    def _extract_yolo_data(r: Any, img_path: str) -> pd.DataFrame:
        """Helper method to extract data from a single YOLO results object into a DataFrame."""
        
        # Check if there are any detections
        n_bxs = r.boxes.data.cpu().shape[0]
        if n_bxs == 0:
            return pd.DataFrame(columns=['Filename', 'names', 'cls', 'x', 'y', 'w', 'h', 'conf', 'imh', 'imw'])
        
        img_name = os.path.basename(img_path).split(".")[0]
        
        # Extract required data
        class_dict = r.names  # Assuming r.names is a dict mapping class indices to names
        cls = r.boxes.cls.data.cpu().numpy().astype(int)
        xywh = r.boxes.xywhn.data.cpu().numpy()
        conf = r.boxes.conf.data.cpu().numpy()
        imh, imw = r.orig_shape[0], r.orig_shape[1]
        
        # Prepare arrays for DataFrame construction
        names = np.array([class_dict[c] for c in cls])
        filenames = np.array([img_name] * n_bxs)
        imhs = np.array([imh] * n_bxs)
        imws = np.array([imw] * n_bxs)
        
        # Combine all data into a single array
        data_array = np.c_[filenames, names, cls, xywh, conf, imhs, imws]
        
        # Create and type-cast the DataFrame
        df = pd.DataFrame(data_array, columns=['Filename', 'names', 'cls', 'x', 'y', 'w', 'h', 'conf', 'imh', 'imw'])
        
        # Apply correct dtypes
        df['cls'] = df['cls'].astype(int)
        df['imh'] = df['imh'].astype(int)
        df['imw'] = df['imw'].astype(int)
        df[['x', 'y', 'w', 'h', 'conf']] = df[['x', 'y', 'w', 'h', 'conf']].astype(float)
        
        return df

    @staticmethod
    def return_lbl_pred_df(results: List[Any], lbls: List[str], imgs: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Processes both YOLO prediction results and corresponding ground truth label files 
        into two structured DataFrames (one for labels, one for predictions).

        Args:
            results (List[Any]): List of raw YOLO results objects (e.g., from YOLOv8).
            lbls (List[str]): List of paths to the ground truth label files (e.g., .txt).
            imgs (List[str]): List of paths to the original images.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (df_lbl, df_prd)
        """
        all_pred_dfs = []
        all_label_dfs = []

        for r, lbl_path, img_path in zip(results, lbls, imgs):
            img_name = os.path.basename(img_path).split(".")[0]
            
            # Use helper to get prediction data
            df_pred_single = YOLODataFormatter._extract_yolo_data(r, img_path)
            all_pred_dfs.append(df_pred_single)
            
            # --- Label Data Processing ---
            try:
                # Assuming YOLO format: class x_c y_c w h
                df_l = pd.read_csv(lbl_path, delimiter=" ", header=None, comment="#")
                df_l.columns = ['cls', 'x', 'y', 'w', 'h']
                
                # Get class names from the prediction object's class dictionary
                class_dict = r.names
                df_l['names'] = df_l['cls'].apply(lambda c: class_dict.get(c, 'unknown'))
                
                df_l['Filename'] = img_name
                df_l = df_l[['Filename', 'names', 'cls', 'x', 'y', 'w', 'h']]
                
                all_label_dfs.append(df_l)
            except pd.errors.EmptyDataError:
                # Handle empty label files gracefully
                continue
            except FileNotFoundError:
                # Handle missing label files
                print(f"Warning: Label file not found for {img_name} at {lbl_path}")
                continue

        # Concatenate all prediction and label DataFrames efficiently
        df_prd = pd.concat(all_pred_dfs, ignore_index=True)
        df_lbl = pd.concat(all_label_dfs, ignore_index=True)
        
        # Add unique identifiers after concatenation
        df_prd['detect_id'] = df_prd['Filename'] + "_dt_" + df_prd.index.astype('str')
        df_lbl['ground_truth_id'] = df_lbl['Filename'] + "_gt_" + df_lbl.index.astype('str')
        
        # Drop duplicates based on the final unique ID (safer post-process check)
        df_prd = df_prd.drop_duplicates(subset="detect_id").reset_index(drop=True)
        
        return df_lbl, df_prd

    @staticmethod
    def return_pred_df(results: List[Any], imgs: List[str]) -> pd.DataFrame:
        """
        Processes YOLO prediction results into a single structured DataFrame 
        without handling ground truth labels. (Similar to a production output class).

        Args:
            results (List[Any]): List of raw YOLO results objects.
            imgs (List[str]): List of paths to the original images.

        Returns:
            pd.DataFrame: DataFrame of predictions (df_prd).
        """
        all_pred_dfs = []

        for r, img_path in zip(results, imgs):
            # Use helper to get prediction data
            df_pred_single = YOLODataFormatter._extract_yolo_data(r, img_path)
            all_pred_dfs.append(df_pred_single)
            
        # Concatenate all prediction DataFrames efficiently
        df_prd = pd.concat(all_pred_dfs, ignore_index=True)

        # Add unique identifier after concatenation
        df_prd['detect_id'] = df_prd['Filename'] + "_dt_" + df_prd.index.astype('str')
        
        # Drop duplicates based on the final unique ID
        df_prd = df_prd.drop_duplicates(subset="detect_id").reset_index(drop=True)
        
        return df_prd