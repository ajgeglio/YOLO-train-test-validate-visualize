import pandas as pd
import numpy as np
import os

'''
The Testing class is designed to handle the output of predictions made by a YOLO model.
It provides methods to process the results, save them to CSV files, and optionally plot the predictions on images.

The difference between this class and the PredictOutput class is that this class is specifically tailored for handling the small dataset testing scenario,
while the PredictOutput class is more focused on returning the results of large-scale predictions runs.

This class also uses advance plotting features to visualize the True and False predictions on images, which is not present in the PredictOutput class.

'''

class Testing:
    @staticmethod
    def return_lbl_pred_df(results, lbls, imgs):
          df_prd = pd.DataFrame(columns=['Filename', 'names','cls','x', 'y', 'w', 'h', 'conf', 'imh', 'imw'])
          df_lbl = pd.DataFrame(columns=['Filename', 'names','cls','x', 'y', 'w', 'h'])
          for r, lbl, img_path in zip(results, lbls, imgs):
                img_name = os.path.basename(img_path).split(".")[0]
                ## indexing the YOLO results object for single image
                n_bxs = r.boxes.data.cpu().shape[0]
                dict = r.names
                cls = r.boxes.cls.data.cpu().numpy().astype(int)
                xywh = r.boxes.xywhn.data.cpu().numpy()
                conf = r.boxes.conf.data.cpu().numpy()
                imh, imw = r.orig_shape[0], r.orig_shape[1]
                names = list(map(lambda x: dict[x], cls))
                img_nm_p = [img_name]*n_bxs
                imh_p = [imh] * n_bxs  # Ensure these are repeated scalar values
                imw_p = [imw] * n_bxs  # Ensure these are repeated scalar values
                ar = np.c_[img_nm_p, names, cls, xywh, conf, imh_p, imw_p]
                df = pd.DataFrame(ar, columns=['Filename', 'names','cls','x', 'y', 'w', 'h', 'conf', 'imh', 'imw'])
                df.x, df.y, df.w, df.h, df.conf = df.x.astype(float), df.y.astype(float), df.w.astype(float), df.h.astype(float), df.conf.astype(float)
                df.imh, df.imw = df.imh.astype(int), df.imw.astype(int)
                
                ## Results Output to dataframe 
                dfs_to_concat = [df_prd, df]
                dfs_to_concat = [d for d in dfs_to_concat if not d.empty]
                df_prd = pd.concat(dfs_to_concat)

                ## Labels Output (_l) suffix
                try:
                     df1 = pd.read_csv(lbl , delimiter=" ", header=None)
                     cls_l, x_l, y_l, w_l, h_l = df1[0], df1[1], df1[2], df1[3], df1[4]
                     names_l = list(map(lambda x: dict[x], cls_l))
                     img_nm_l = [img_name]*len(cls_l)
                     ar_l = np.c_[img_nm_l, names_l, cls_l, x_l, y_l, w_l, h_l]
                     df_l = pd.DataFrame(ar_l, columns = ['Filename', 'names','cls','x', 'y', 'w', 'h'])
                     df_l.x, df_l.y, df_l.w, df_l.h = df_l.x.astype(float), df_l.y.astype(float), df_l.w.astype(float), df_l.h.astype(float)
                     
                     dfs_to_concat_lbl = [df_lbl, df_l]
                     dfs_to_concat_lbl = [d for d in dfs_to_concat_lbl if not d.empty]
                     df_lbl = pd.concat(dfs_to_concat_lbl)
                except pd.errors.EmptyDataError:
                     pass
                df_prd['detect_id'] = df_prd['Filename'].apply(lambda x: x.split(".")[0]) +"_dt_"+ df_prd.index.astype('str')
                df_lbl['ground_truth_id'] = df_lbl['Filename'].apply(lambda x: x.split(".")[0]) +"_"+ df_lbl.index.astype('str')
          return df_lbl, df_prd
    @staticmethod
    def return_pred_df(results, imgs):
        df_prd = pd.DataFrame(columns=['Filename', 'names','cls','x', 'y', 'w', 'h', 'conf', 'imh', 'imw'])
        for r, img_path in zip(results, imgs):
            img_name = os.path.basename(img_path).split(".")[0]
            ## indexing the YOLO results object for single image
            n_bxs = r.boxes.data.cpu().shape[0]
            dict = r.names
            cls = r.boxes.cls.data.cpu().numpy().astype(int)
            xywh = r.boxes.xywhn.data.cpu().numpy() #https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes
            ## optional output for box xywh in absolute pixel coordinates
            # xywh = r.boxes.xywh.data.cpu().numpy() #https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Boxes
            ## optional output for top left, bottom right box box coodinates
            # x1y1x2y2 = ultralytics.utils.ops.xywh2xyxy(xywh) #https://docs.ultralytics.com/reference/utils/ops/#ultralytics.utils.ops.xyxy2xywh
            conf = r.boxes.conf.data.cpu().numpy()
            imh, imw = [r.orig_shape[0]]*n_bxs, [r.orig_shape[1]]*n_bxs
            names = list(map(lambda x: dict[x] , cls))#converting int labels to names from a dictionary
            img_nm_p = [img_name]*n_bxs

            ## Results Output to dataframe 
            ar = np.c_[img_nm_p, names, cls, xywh, conf, imh, imw]
            df = pd.DataFrame(ar, columns=['Filename', 'names','cls','x', 'y', 'w', 'h', 'conf', 'imh', 'imw'])
            df.x, df.y, df.w, df.h, df.conf, df.imh, df.imw = df.x.astype(float), df.y.astype(float), df.w.astype(float), df.h.astype(float), df.conf.astype(float), df.imh.astype(int), df.imw.astype(int)
            
            dfs_to_concat = [df_prd, df]
            dfs_to_concat = [d for d in dfs_to_concat if not d.empty]
            df_prd = pd.concat(dfs_to_concat)

        ## add unique identifier to each goby detect
        df['detect_id'] = df['Filename'].apply(lambda x: x.split(".")[0]) +"_dt_"+ df.index.astype('str')
        ## drop duplicates in the master. Necessary if the inference was interrutpted
        df = df.drop_duplicates(subset="detect_id")
        return df