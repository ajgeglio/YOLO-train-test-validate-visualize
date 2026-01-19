import os
import glob
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from typing import Any, List, Optional, Dict, Tuple
# Assuming YOLODataFormatter (the refactored Testing class) is accessible
from dataFormatter import YOLODataFormatter 
# Assuming CalculateIntersection is accessible
from calculateIOU import CalculateIntersection 

class BatchOutputProcessor:
    """
    Handles the output of large-scale YOLO inference runs. 
    Provides methods to process results, incrementally append them to CSV files, 
    and plot predictions.
    """
    
    def __init__(self) -> None:
        pass # Explicitly empty init

    # --- Helper Functions (Refactored to be Internal) ---
    @staticmethod
    def _write_labels(lbl_path: str, class_dict: Dict[int, str], img_id: str, im_h: int, im_w: int, lbls_csv_pth: str) -> Optional[pd.DataFrame]:
        """Helper to read label info and append it to the labels CSV."""
        try:
            # Read YOLO format: cls x_c y_c w h
            df_l = pd.read_csv(lbl_path, delimiter=" ", header=None, comment="#")
            df_l.columns = [0, 1, 2, 3, 4] # Temp columns
            
            # Map class IDs to names
            cls_l, x_l, y_l, w_l, h_l = df_l[0], df_l[1], df_l[2], df_l[3], df_l[4]
            names_l = list(map(lambda x: class_dict[x], cls_l))
            
            # Prepare arrays for writing
            n_bxs = len(cls_l)
            img_nm_l = [img_id] * n_bxs
            im_h_l, im_w_l = [im_h] * n_bxs, [im_w] * n_bxs
            
            # Combine all label data
            ar_l = np.c_[img_nm_l, names_l, cls_l, x_l, y_l, w_l, h_l, im_h_l, im_w_l]
            df_write = pd.DataFrame(ar_l)
            df_write = df_write.sort_index(axis=0) # Sort by index for consistency
            # Append data to the CSV file without header
            df_write.to_csv(lbls_csv_pth, mode='a', header=False)
            
            # Return original structure (used for plotting)
            return df_l 
        except (pd.errors.EmptyDataError, FileNotFoundError):
            return None

    @staticmethod
    def _plot_predictions(r: Any, img_name: str, plots_folder: str, df_labels: Optional[pd.DataFrame], im_h: int, im_w: int, has_labels: bool):
        """Helper to plot YOLO predictions and optionally ground truth labels."""
        
        # 1. Plot YOLO predictions using the built-in 'r.plot()'
        # Note: r.plot() handles the prediction boxes automatically
        im_array = r.plot(conf=True, probs=False, line_width=1, labels=True, font_size=4)
        im = Image.fromarray(im_array[..., ::-1])
        draw = ImageDraw.Draw(im)
        
        # 2. Add ground truth labels (YOLO format: x_c, y_c, w, h)
        if has_labels and df_labels is not None:
            # df_labels columns are 0, 1, 2, 3, 4 (cls, x_c, y_c, w, h)
            for _, row in df_labels.iterrows():
                try:
                    # Normalized coordinates
                    x_n, y_n, w_n, h_n = float(row[1]), float(row[2]), float(row[3]), float(row[4])
                    
                    # Convert to absolute pixel coordinates
                    x = x_n * im_w
                    y = y_n * im_h
                    w = w_n * im_w
                    h = h_n * im_h
                    
                    # Convert to top-left, bottom-right coordinates
                    x1 = x - w/2
                    y1 = y - h/2
                    x2 = x + w/2
                    y2 = y + h/2
                    
                    draw.rectangle((x1, y1, x2, y2), outline="#FF00FF", width=2)
                except Exception:
                    continue 

        # 3. Save the final image
        img_save_path = os.path.join(plots_folder, img_name)
        im.save(img_save_path)

    # --- Core Processing Methods ---

    @staticmethod
    def YOLO_predict_w_outut(r: Any, lbl_path: str, img_path: str, pred_csv_pth: str, lbls_csv_pth: str, plots_folder: Optional[str]=None, plot: bool=False, has_labels: bool=False):
        """
        Processes a single YOLO prediction result (standard bounding box) and appends 
        it to CSV files. Uses the shared prediction extraction logic.
        """
        img_name = os.path.basename(img_path)
        img_id = img_name.split(".")[0]
        
        # 1. Extract Prediction Data (REFACTOR: Used shared helper method)
        df_pred_single = YOLODataFormatter._extract_yolo_data(r, img_path)
        
        # Determine image dimensions (essential for plotting/labels even if no detections)
        im_h, im_w = r.orig_shape[0], r.orig_shape[1]
        df_labels = None
        
        if not df_pred_single.empty:
            # Write prediction data to CSV
            # Data types were already set correctly in _extract_yolo_data
            df_pred_single.to_csv(pred_csv_pth, mode='a', header=False)
            
        # 2. Process and Write Labels
        if has_labels:
            df_labels = BatchOutputProcessor._write_labels(lbl_path, r.names, img_id, im_h, im_w, lbls_csv_pth)
        
        # 3. Plotting
        if plot and plots_folder is not None:
            BatchOutputProcessor._plot_predictions(r, img_name, plots_folder, df_labels, im_h, im_w, has_labels)
        
    @staticmethod
    def YOLO_predict_w_outut_sahi(coco_json_list: List[Dict], lbl_path: str, img_path: str, pred_csv_pth: str, lbls_csv_pth: str, plots_folder: Optional[str]=None, plot: bool=False, has_labels: bool=False):
        """
        Processes predictions from SAHI (or similar COCO-format) output and appends 
        it to CSV files.
        """
        img_name = os.path.basename(img_path)
        img_id = img_name.split(".")[0]

        rows = []
        im_w, im_h = None, None
        
        # 1. Parse COCO/SAHI JSON list
        for idx, obj in enumerate(coco_json_list):
            bbox = obj['bbox']  # [x, y, w, h] in absolute pixels
            category_id = obj['category_id']
            score = obj.get('score', 1.0)
            
            if idx == 0:
                try:
                    from PIL import Image
                    im = Image.open(img_path)
                    im_w, im_h = im.size
                except Exception:
                    im_w, im_h = 1, 1  # fallback
            
            # Convert bbox to normalized xywh
            x, y, w, h = bbox
            x_c = (x + w / 2) / im_w
            y_c = (y + h / 2) / im_h
            w_n = w / im_w
            h_n = h / im_h
            
            rows.append([
                img_id,
                obj.get('category_name', str(category_id)),
                category_id,
                x_c, y_c, w_n, h_n,
                score,
                im_h, im_w
            ])
            
        # 2. Write Prediction Data
        if rows:
            df = pd.DataFrame(rows)
            # Use original columns for consistency with YOLO_predict_w_outut output
            df.columns = ['Filename', 'names', 'cls', 'x', 'y', 'w', 'h', 'conf', 'imh', 'imw'] 
            df.to_csv(pred_csv_pth, mode='a', header=False, index=False)
            
        # 3. Process and Write Labels
        df_labels = None
        if has_labels:
            # We don't have r.names here, so a placeholder dict is necessary if using _write_labels
            temp_dict = {row[2]: row[1] for row in rows} if rows else {}
            df_labels = BatchOutputProcessor._write_labels(lbl_path, temp_dict, img_id, im_h, im_w, lbls_csv_pth)
            
        # 4. Plotting (SAHI-specific plotting retained)
        if plot and rows:
            try:
                im = Image.open(img_path).convert("RGB")
                draw = ImageDraw.Draw(im)
                
                # Plot predictions (red outline)
                for row in rows:
                    # row: [img_id, name, cls, x_c, y_c, w_n, h_n, conf, im_h, im_w]
                    x_c, y_c, w_n, h_n = float(row[3]), float(row[4]), float(row[5]), float(row[6])
                    im_w, im_h = int(row[9]), int(row[8])
                    x = x_c * im_w; y = y_c * im_h
                    w = w_n * im_w; h = h_n * im_h
                    x1 = x - w/2; y1 = y - h/2
                    x2 = x + w/2; y2 = y + h/2
                    draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)
                    
                # Plot labels (black outline)
                if has_labels and df_labels is not None:
                    # df_labels columns are 0, 1, 2, 3, 4 (cls, x_c, y_c, w, h)
                    for _, row in df_labels.iterrows():
                        try:
                            x, y, w, h = float(row[1])*im_w, float(row[2])*im_h, float(row[3])*im_w, float(row[4])*im_h
                        except Exception:
                            continue
                        x1 = x - w/2; y1 = y - h/2
                        x2 = x + w/2; y2 = y + h/2
                        draw.rectangle((x1, y1, x2, y2), outline=(0, 0, 0), width=4)
                        
                img_save_path = os.path.join(plots_folder, img_name)
                im.save(img_save_path)
            except Exception as e:
                print(f"Plotting failed for {img_path}: {e}")

    @staticmethod
    def YOLO_predict_w_outut_obb(r: Any, lbl_path: str, img_path: str, pred_csv_pth: str, lbls_csv_pth: str, plots_folder: Optional[str]=None, plot: bool=False, has_labels: bool=False):
        """
        Processes a single YOLO prediction result (Oriented Bounding Box - OBB) 
        and appends it to CSV files.
        """
        img_name = os.path.basename(img_path)
        img_id = img_name.split(".")[0]
        
        n_bxs = r.obb.data.cpu().shape[0]
        dict = r.names
        
        # 1. Extract OBB Data
        cls = r.obb.cls.data.cpu().numpy().astype(int)
        xyxyxyxy = [x.flatten() for x in r.obb.xyxyxyxyn.data.cpu().numpy()] # Normalized corner points
        xywhr = r.obb.xywhr.data.cpu().numpy() # Normalized center, width, height, rotation
        conf = r.obb.conf.data.cpu().numpy()
        
        im_h, im_w = r.orig_shape[0], r.orig_shape[1]
        im_h_p, im_w_p = [im_h]*n_bxs, [im_w]*n_bxs
        names = list(map(lambda x: dict[x], cls))
        img_nm_p = [img_id]*len(cls)

        # 2. Write Prediction Data
        ar = np.c_[img_nm_p, names, cls, xyxyxyxy, xywhr, conf, im_h_p, im_w_p]
        df = pd.DataFrame(ar)
        df.to_csv(pred_csv_pth, mode='a', header=False, index=False)
        
        # 3. Process and Write Labels
        df_labels = None
        if has_labels:
            try:
                df_labels = pd.read_csv(lbl_path, delimiter=" ", header=None)
                # OBB labels typically have 9 columns: cls x1 y1 x2 y2 x3 y3 x4 y4
                cls_l, x1_l, y1_l, x2_l, y2_l, x3_l, y3_l, x4_l, y4_l = df_labels[0], df_labels[1], df_labels[2], df_labels[3], df_labels[4], df_labels[5], df_labels[6], df_labels[7], df_labels[8]
                
                # Assuming class dictionary needs to be defined for OBB labels here or passed in
                cls_dict = {0: "goby", 1: "alewife", 3: "notfish", 4: "other"}
                names_l = list(map(lambda x: cls_dict.get(x, str(x)), cls_l))
                
                img_nm_l = [img_id]*len(cls_l)
                ar_l = np.c_[img_nm_l, names_l, cls_l, x1_l, y1_l, x2_l, y2_l, x3_l, y3_l, x4_l, y4_l]
                df_l = pd.DataFrame(ar_l)
                df_l.to_csv(lbls_csv_pth, mode='a', header=False, index=False)
            except (pd.errors.EmptyDataError, FileNotFoundError):
                pass

        # 4. Plotting (OBB-specific plotting retained)
        if plot and n_bxs > 0:
            img_save_path = os.path.join(plots_folder,img_name)
            # r.plot() handles the predicted OBB boxes
            im_array = r.plot(conf=True, probs=False, line_width=1, labels=True, font_size=4)
            im = Image.fromarray(im_array[..., ::-1])
            draw = ImageDraw.Draw(im)
            
            if has_labels and df_labels is not None:
                # df_labels columns are 0, 1..8 (cls, x1, y1, x2, y2, x3, y3, x4, y4)
                for _, row in df_labels.iterrows():
                    try:
                        # Normalized coordinates
                        xs = (row[1], row[3], row[5], row[7])
                        ys = (row[2], row[4], row[6], row[8])
                        # Convert to absolute pixel coordinates
                        x1, x2, x3, x4 = (float(x)*im_w for x in xs)
                        y1, y2, y3, y4 = (float(y)*im_h for y in ys)
                        
                        draw.polygon([x1,y1,x2,y2, x3, y3, x4, y4], outline=(0, 0, 0), width=3)
                    except Exception:
                        continue
            im.save(img_save_path)
    
    # --- Utility and Intersection Methods ---

    def process_labels(self, label_list: List[str]) -> pd.DataFrame:
        """
        Combines YOLO-format label files into a single DataFrame.
        Refactored to collect rows before final concatenation for efficiency.
        """
        rows = []
        
        for label_path in label_list:
            Filename = os.path.basename(label_path).split(".")[0]
            try:
                # Read YOLO format: cls x_c y_c w h
                df_l = pd.read_csv(label_path, delimiter=" ", header=None, comment="#")
                df_l.columns = ["cls", "x", "y", "w", "h"]
                df_l['Filename'] = Filename
                
                # Append rows to list
                rows.extend(df_l.to_dict('records'))
                
            except (pd.errors.EmptyDataError, FileNotFoundError):
                continue
            except Exception as e:
                print(f"Error processing label file {label_path}: {e}")
                continue
                
        # Final efficient concatenation
        labels_df = pd.DataFrame(rows)
        
        if not labels_df.empty:
            # Ensure correct types for merging/calculation
            labels_df[['cls', 'x', 'y', 'w', 'h']] = labels_df[['cls', 'x', 'y', 'w', 'h']].astype({
                'cls': object, 
                'x': float, 'y': float, 'w': float, 'h': float
            })
            
        return labels_df[['Filename', 'cls', 'x', 'y', 'w', 'h']]

    def intersection_df(self, fish_box_df: pd.DataFrame, cage_box_df: pd.DataFrame, input_type: str) -> pd.DataFrame:
        """
        Calculate the intersection (specifically IOU/containment for this logic) 
        between fish bounding boxes and cage bounding boxes.
        """
        # Merge fish and cage bounding boxes on "Filename"
        df_all = pd.merge(fish_box_df, cage_box_df, on="Filename", how="left", suffixes=("_f", "_c"))
        assert len(df_all) == len(fish_box_df), "Lengths do not match after merging."

        # Add additional columns based on input type
        if input_type == "ground_truth":
            # Assuming ground truth comes from YoloDataFormatter and needs imh/imw added back
            # The structure relies on the imh/imw columns being present for intersection calculation
            # This is a potential point of failure if the original labels didn't include imh/imw
            # Assuming imh/imw are present in fish_box_df from previous steps or merging is robust
            df_all = df_all.rename(columns={"imw_l":"imw", "imh_l":"imh"}, errors='ignore') # Added errors='ignore'
            df_all["conf"] = 1.0

        # Calculate intersection area (relies on CalculateIntersection().get_intersection(row)
        df_all['intersection'] = df_all.apply(
            lambda row: CalculateIntersection().get_intersection(row) if not row[['x_c', 'y_c', 'w_c', 'h_c']].isnull().any() else np.nan,
            axis=1
        )

        # Determine if the fish box is inside the cage box (using a 0.5 threshold)
        df_all["inside"] = np.where(df_all.intersection > 0.5, 1, 0)

        # Select relevant columns for output
        columns = [
            'Filename', 'cls_f', 'x_f', 'y_f', 'w_f', 'h_f',
            'x_c', 'y_c', 'w_c', 'h_c', 'imh', 'imw',
            f'{input_type}_id', 'intersection', 'inside', 'conf'
        ]
        columns = [col for col in columns if col in df_all.columns]
        return df_all[columns]

    def save_cage_box_label_outut(self, df_pred: pd.DataFrame, df_lbl: Optional[pd.DataFrame], img_dir: str, run_path: str):
        """
        Calculates and saves the intersection analysis (detections and labels vs. cage boxes).
        """
        # Get cage bounding box files
        cage_box_dir = os.path.join(os.path.dirname(img_dir), "cages")
        cage_box_list = sorted(glob.glob(os.path.join(cage_box_dir, "*.txt")))
        
        if not cage_box_list:
            print(f"No cage bounding box files found in {cage_box_dir}")
            return 
        
        # Process the cage bounding box files into a dataframe
        cage_box_df = self.process_labels(cage_box_list)
        output_name = os.path.basename(run_path)
        
        # 1. Prediction Analysis
        prediction_box_df = df_pred.copy()
        df_gopro_prediction_analysis = self.intersection_df(prediction_box_df, cage_box_df, input_type="detect")
        df_gopro_prediction_analysis.to_csv(os.path.join(run_path, f"{output_name}_cage_predictions.csv"), index=False)

        # 2. Label Analysis
        if df_lbl is not None:
            lbl_df = df_lbl.copy()
            df_gopro_label_analysis = self.intersection_df(lbl_df, cage_box_df, input_type="ground_truth")
            df_gopro_label_analysis.to_csv(os.path.join(run_path, f"{output_name}_cage_labels.csv"), index=False)

        print(f"Output with cages saved to {run_path}")