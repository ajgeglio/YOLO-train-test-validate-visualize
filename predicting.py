import os
import glob
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from utils import Utils
from iou import CalculateIntersection

'''
The PredictOutput class is designed to handle the output of predictions made by a YOLO model.
It provides methods to process the results, save them to CSV files, and optionally plot the predictions on images.

The difference between this class and the Testing class is that this class is specifically tailored for handling the output of large-scale inference runs,
while the Testing class is more focused on returning dataframes for evaluation purposes.

This class will append the results to existing CSV files, allowing for incremental updates without overwriting previous data.

'''
class PredictOutput:
    def __init__(self) -> None:
        self

    def YOLO_predict_w_outut(r, lbl, img_path, pred_csv_pth, lbls_csv_pth, plots_folder=None, plot=False, has_labels=False):
        img_name = os.path.basename(img_path)
        img_id = img_name.split(".")[0]
        ## indexing the YOLO results object for single image
        n_bxs = r.boxes.data.cpu().shape[0]
        dict = r.names
        cls = r.boxes.cls.data.cpu().numpy().astype(int)
        xywh = r.boxes.xywhn.data.cpu().numpy()
        conf = r.boxes.conf.data.cpu().numpy()
        # # optional output for box xywh in absolute pixel coordinates
        # xywh = r.boxes.xywh.data.cpu().numpy() #https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes
        # # optional output for top left, bottom right box box coodinates
        # x1y1x2y2 = ultralytics.utils.ops.xywh2xyxy(xywh) #https://docs.ultralytics.com/reference/utils/ops/#ultralytics.utils.ops.xyxy2xywh
        im_h, im_w = r.orig_shape[0], r.orig_shape[1]
        im_h_p, im_w_p = [im_h]*n_bxs, [im_w]*n_bxs
        names = list(map(lambda x: dict[x], cls))
        img_nm_p = [img_id]*len(cls)

        ## Results Output to dataframe 
        ar = np.c_[img_nm_p, names, cls, xywh, conf, im_h_p, im_w_p]
        df = pd.DataFrame(ar)
        df.to_csv(pred_csv_pth, mode='a', header=False)
        
        ## Labels Output (_l) suffix
        if has_labels:
            try:
                df1 = pd.read_csv(lbl , delimiter=" ", header=None)
                cls_l, x_l, y_l, w_l, h_l = df1[0], df1[1], df1[2], df1[3], df1[4]
                names_l = list(map(lambda x: dict[x], cls_l))
                img_nm_l = [img_id]*len(cls_l)
                im_h_l, im_w_l = [im_h]*len(cls_l), [im_w]*len(cls_l)
                ar_l = np.c_[img_nm_l, names_l, cls_l, x_l, y_l, w_l, h_l, im_h_l, im_w_l]
                df_l = pd.DataFrame(ar_l)
                df_l.to_csv(lbls_csv_pth, mode='a', header=False)
            except pd.errors.EmptyDataError:
                pass
        
        if plot:
            if n_bxs > 0:
                ##### DRAWING #####
                ## Drawing the prediction
                img_save_path = os.path.join(plots_folder,img_name)
                im_array = r.plot(conf=True, probs=False, line_width=1, labels=True, font_size=4)  # plot a BGR numpy array of predictions
                ## Drawing the label in black
                im_h, im_w = im_array.shape[0], im_array.shape[1]
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                draw = ImageDraw.Draw(im)
                if has_labels:
                    for index, row in df1.iterrows():
                        x, y, w, h = row[1]*im_w, row[2]*im_h, row[3]*im_w, row[4]*im_h
                        x1 = x - w/2
                        y1 = y - h/2
                        x2 = x + w/2
                        y2 = y + h/2
                        draw.rectangle((x1,y1,x2,y2), outline=(0, 0, 0), width=4)
                im.save(img_save_path)  # save image
                
    def YOLO_predict_w_outut_obb(r, lbl, img_path, pred_csv_pth, lbls_csv_pth, plots_folder=None, plot=False, has_labels=False):
        img_name = os.path.basename(img_path)
        img_id = img_name.split(".")[0]
        ## indexing the YOLO results object for single image
        n_bxs = r.obb.data.cpu().shape[0]
        dict = r.names
        cls = r.obb.cls.data.cpu().numpy().astype(int)
        xyxyxyxy = r.obb.xyxyxyxyn.data.cpu().numpy()
        xyxyxyxy = [x.flatten() for x in xyxyxyxy]
        xywhr = r.obb.xywhr.data.cpu().numpy()
        conf = r.obb.conf.data.cpu().numpy()
        # # optional output for box xywh in absolute pixel coordinates
        # xywh = r.obb.xywh.data.cpu().numpy() #https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes
        # # optional output for top left, bottom right box box coodinates
        # x1y1x2y2 = ultralytics.utils.ops.xywh2xyxy(xywh) #https://docs.ultralytics.com/reference/utils/ops/#ultralytics.utils.ops.xyxy2xywh
        im_h, im_w = r.orig_shape[0], r.orig_shape[1]
        im_h_p, im_w_p = [im_h]*n_bxs, [im_w]*n_bxs
        names = list(map(lambda x: dict[x], cls))
        img_nm_p = [img_id]*len(cls)

        ## Results Output to dataframe 
        ar = np.c_[img_nm_p, names, cls, xyxyxyxy, xywhr, conf, im_h_p, im_w_p]
        df = pd.DataFrame(ar)
        df.to_csv(pred_csv_pth, mode='a', header=False)
        
        ## Labels Output (_l) suffix
        if has_labels:
            try:
                df1 = pd.read_csv(lbl , delimiter=" ", header=None)
                cls_l, x1_l, y1_l, x2_l, y2_l, x3_l, y3_l, x4_l, y4_l = df1[0], df1[1], df1[2], df1[3], df1[4], df1[5], df1[6], df1[7], df1[8]
                cls_dict = {0: "goby", 1: "alewife", 3: "notfish", 4: "other"}
                names_l = list(map(lambda x: cls_dict[x], cls_l))
                img_nm_l = [img_id]*len(cls_l)
                ar_l = np.c_[img_nm_l, names_l, cls_l, x1_l, y1_l, x2_l, y2_l, x3_l, y3_l, x4_l, y4_l]
                df_l = pd.DataFrame(ar_l)
                df_l.to_csv(lbls_csv_pth, mode='a', header=False)
            except pd.errors.EmptyDataError:
                pass
            except FileNotFoundError:
                pass
        
        if plot:
            if n_bxs > 0:
                ##### DRAWING #####
                ## Drawing the prediction
                img_save_path = os.path.join(plots_folder,img_name)
                im_array = r.plot(conf=True, probs=False, line_width=1, labels=True, font_size=4)  # plot a BGR numpy array of predictions
                ## Drawing the label in black
                im_h, im_w = im_array.shape[0], im_array.shape[1]
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                draw = ImageDraw.Draw(im)
                if has_labels:
                    for index, row in df1.iterrows():
                        # x, y, w, h = row[1]*im_w, row[2]*im_h, row[3]*im_w, row[4]*im_h
                        xs = (row[1], row[3], row[5], row[7])
                        print(xs)
                        ys = (row[2], row[4], row[6], row[8])
                        print(ys)
                        x1, x2, x3, x4 = (x*im_w for x in xs)
                        y1, y2, y3, y4 = (y*im_h for y in ys)
                        draw.polygon([x1,y1,x2,y2, x3, y3, x4, y4], outline=(0, 0, 0), width=3)
                im.save(img_save_path)  # save image

    def process_labels(self, label_list):
    # function to combine the the labels and cage boxes for LMBS images
        labels_df = pd.DataFrame(columns=["Filename", "cls", "x", "y", "w", "h"])
        for label in label_list:
            Filename = os.path.basename(label).split(".")[0]
            # read the label file
            with open(label, "r") as f:
                lines = f.readlines()
            # Skip processing if the file is empty
            if not lines:
                continue
            # extract the name and coordinates
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    name, x, y, w, h = parts
                    # append to the dataframe
                    labels_df = pd.concat(
                        [labels_df, pd.DataFrame([{"Filename": Filename, "cls": name, "x": float(x), "y": float(y), "w": float(w), "h": float(h)}])],
                        ignore_index=True
                    )
        return labels_df

    def intersection_df(self, fish_box_df, cage_box_df, input_type="detect"):
        """
        Calculate the intersection between fish bounding boxes and cage bounding boxes.

        Args:
            fish_box_df (pd.DataFrame): DataFrame containing fish bounding boxes.
            cage_box_df (pd.DataFrame): DataFrame containing cage bounding boxes.
            input_type (str): Type of input, either "bboxes", "labels", or "need_id".

        Returns:
            pd.DataFrame: DataFrame with intersection and inside calculations.
        """
        # Merge fish and cage bounding boxes on "Filename"
        df_all = pd.merge(fish_box_df, cage_box_df, on="Filename", how="left", suffixes=("_f", "_c"))
        assert len(df_all) == len(fish_box_df), "Lengths do not match after merging."

        # Add additional columns based on input type
        if input_type == "ground_truth":
            df_all = df_all.rename(columns={"imw_l":"imw", "imh_l":"imh"})
            df_all["conf"] = 1.0

        # Calculate intersection area
        df_all['intersection'] = df_all.apply(
            lambda row: CalculateIntersection().get_intersection(row) if not row.isnull().any() else np.nan,
            axis=1
        )

        # Determine if the fish box is inside the cage box
        df_all["inside"] = np.where(df_all.intersection > 0.5, 1, 0)

        # Select relevant columns for output
        columns = [
            'Filename', 'cls_f', 'x_f', 'y_f', 'w_f', 'h_f',
            'x_c', 'y_c', 'w_c', 'h_c', 'imh', 'imw',
            f'{input_type}_id', 'intersection', 'inside', 'conf'
        ]
        return df_all[columns]

    def save_cage_box_label_outut(self, df_pred, df_lbl, img_dir, run_path):
        
        # Get a sorted list of all cage bounding box files in the directory
        cage_box_dir = os.path.join(os.path.dirname(img_dir), "cages")
        cage_box_list = sorted(glob.glob(os.path.join(cage_box_dir, "*.txt")))
        if len(cage_box_list) == 0:
            print(f"No cage bounding box files found in {cage_box_dir}")
        
        # Process the cage bounding box files into a dataframe
        cage_box_df = self.process_labels(cage_box_list)
        output_name = os.path.basename(run_path)
        # Process the prediction output
        prediction_box_df = df_pred.copy()
        # Perform the intersection analysis for the detections, and save the results to a CSV file
        df_gopro_prediction_analysis = self.intersection_df(prediction_box_df, cage_box_df, input_type="detect")
        df_gopro_prediction_analysis.to_csv(os.path.join(run_path, f"{output_name}_cage_predictions.csv"), index=False)

        if df_lbl is not None:
            # Get a sorted list of all label files in the directory
            # bbox_list = sorted(glob.glob(os.path.join(root, "labels", "*.txt")))
            # # Process the label files into a dataframe
            # label_box_df = self.process_labels(bbox_list)      
            lbl_df = df_lbl.copy()
            # Perform the intersection analysis for the annotated labels, and save the results to a CSV file
            df_gopro_label_analysis = self.intersection_df(lbl_df, cage_box_df, input_type="ground_truth")
            df_gopro_label_analysis.to_csv(os.path.join(run_path, f"{output_name}_cage_labels.csv"), index=False)

        print(f"Output with cages saved to {run_path}")