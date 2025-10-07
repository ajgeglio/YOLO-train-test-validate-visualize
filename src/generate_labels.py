import os
import json
import cv2
import numpy as np
import pandas as pd

class GenerateLabels:
    def __init__(self, img_pth_lst=None, jsn_pth_lst=None, lbl_msk_pths=None, bbox_pths = None, mer_pths = None):
        self.img_pth_lst = img_pth_lst
        self.jsn_pth_lst = jsn_pth_lst
        self.lbl_msk_pths = lbl_msk_pths
        self.bbox_pths = bbox_pths
        self.mer_pths = mer_pths

    def json_to_img_lbls(self, save_path):
        # generates img_name.txt label file for every image with a coco json label  
        assert len(self.img_pth_lst) == len(self.jsn_pth_lst)
        lbl_len = len(self.jsn_pth_lst)
        for i in range(lbl_len):
            jsonfile = self.jsn_pth_lst[i]
            jf = open(jsonfile)
            data = json.load(jf)
            try:
                img_name = data['image']['filename']
                img_path = self.img_pth_lst[i]
                imh, imw = self.get_shape_pil(img_path)
                label_list = []
                for l in range(len(data['annotations'])): # for each img there may be multiple objects
                    try:
                        if data['annotations'][l]['name'] == 'Fish':
                            x = data['annotations'][l]['bounding_box']['x']
                            y = data['annotations'][l]['bounding_box']['y']
                            w = data['annotations'][l]['bounding_box']['w']
                            h = data['annotations'][l]['bounding_box']['h']
                            label_list.extend("0 ")
                            # label_list.extend(f"{data['annotations'][l]['name']} ")
                            label_list.extend(f"{(x+w/2)/imw} ")
                            label_list.extend(f"{(y+h/2)/imh} ")
                            label_list.extend(f"{(w)/imw} ")
                            label_list.extend(f"{(h)/imh}\n")
                    except: pass
            except: pass
            if len(set(label_list))>=5:
                print(f"generating label file in {save_path} {i+1}/{lbl_len}", end=' \r')
                with open(f"{save_path}/{str(img_name).split('.')[0]}.txt", 'w') as f:
                    f.writelines(label_list)
                    f.close()
            else: 
                print(f"generating label file in {save_path} {i+1}/{lbl_len}", end=' \r')
                with open(f"{save_path}/{str(img_name).split('.')[0]}.txt", 'w') as f:
                    f.writelines("")
                    f.close()

    def lbl_masks_to_img_lbls(self, save_path, color=None):
        # generates img_name.txt label file for every image with a mask
        # This is for the various colored fish masks overlayed on an image
        assert len(self.img_pth_lst) == len(self.lbl_msk_pths)
        lbl_len = len(self.lbl_msk_pths)
        for i in range(lbl_len):
            msk_path = self.lbl_msk_pths[i]
            img_path = self.img_pth_lst[i]
            img_name = os.path.basename(img_path)
            msk_array = cv2.imread(msk_path)
            # This is for the pink fish masks overlayed on an image
            if color == "pink":
                mask = cv2.inRange(msk_array, (140,20,165), (227,20,227))
            # This is for the green fish masks overlayed on an image which happen to be BGR
            elif color == "green":
                mask = cv2.inRange(msk_array, (40,225,40), (60,255,60))
            elif color == None:
                mask = cv2.inRange(msk_array, (10,10,10), (255,255,255))
            imh, imw = msk_array.shape[0], msk_array.shape[1]
            # threshold based on a binary threshold from the mask
            ret, thresh = cv2.threshold(mask, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # img1 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # array of bounding boxes filtered out small boxes
            ar = [cv2.boundingRect(cnt) for cnt in contours if cv2.boundingRect(cnt)[-1]*cv2.boundingRect(cnt)[-2]>100]
            cls_ = [0]*len(ar)
            df = pd.DataFrame(np.c_[cls_,ar])
            if not df.empty:
                x, y, w, h = df[1], df[2], df[3], df[4]
                df[1] = (x+w/2)/imw
                df[2] = (y+h/2)/imh
                df[3] = (w)/imw
                df[4] = (h)/imh
                print(f"generating label file from mask in labels_from_masks {i+1}/{lbl_len}", end='    \r')
                df.to_csv(f"{save_path}/{str(img_name).split('.')[0]}.txt", sep=' ', header=False, index=False)
            else: 
                df = pd.DataFrame()
                df.to_csv(f"{save_path}/{str(img_name).split('.')[0]}.txt", sep=' ', header=False, index=False)

    def lbl_masks_to_contours(self, save_path, color=None):
        assert len(self.img_pth_lst) == len(self.lbl_msk_pths)
        lbl_len = len(self.lbl_msk_pths)
        for i in range(lbl_len):
            msk_path = self.lbl_msk_pths[i]
            img_path = self.img_pth_lst[i]
            img_name = os.path.basename(img_path)
            msk_array = cv2.imread(msk_path)
            # This is for the pink fish masks overlayed on an image
            if color == "pink":
                mask = cv2.inRange(msk_array, (140,20,165), (227,20,227))
            # This is for the green fish masks overlayed on an image which happen to be BGR
            elif color == "green":
                mask = cv2.inRange(msk_array, (40,225,40), (60,255,60))
            elif color == None:
                mask = cv2.inRange(msk_array, (10,10,10), (255,255,255))
            imh, imw = msk_array.shape[0], msk_array.shape[1]
            # threshold based on a binary threshold from the mask
            ret, thresh = cv2.threshold(mask, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ar = [np.insert(cnt.flatten(), 0, 0) for cnt in contours]
            # cls_ = [0]*len(ar)
            df = pd.DataFrame(ar)              
            if not df.empty:
                print(f"generating contours from mask in labels_from_masks {i+1}/{lbl_len}", end='    \r')
                df.to_csv(f"./{save_path}/{str(img_name).split('.')[0]}.txt", sep=' ', header=False, index=False)
            else: 
                print(f"generating blanklbl from mask in labels_from_masks {i+1}/{lbl_len}", end='    \r')
                df = pd.DataFrame()
                df.to_csv(f"./{save_path}/{str(img_name).split('.')[0]}.txt", sep=' ', header=False, index=False)  

    @staticmethod
    def make_master_lbl_df_gopro(lbl_path_lst):
        lbl_lst = []
        for lbl_file in lbl_path_lst:
            lbl = pd.read_csv(lbl_file, sep=' ', names=["cls","x","y","w","h"], index_col=None)
            image_id = os.path.basename(lbl_file).split(".")[0]
            lbl['Filename'] = image_id
            lbl['id'] = lbl.index.astype(int)
            lbl_lst.append(lbl)
        master_lbl_df = pd.concat(lbl_lst, ignore_index=True)
        master_lbl_df['fish_id'] = master_lbl_df.Filename.astype(str) +"_"+ master_lbl_df.id.astype(str)
        master_lbl_df = master_lbl_df[["Filename", "id", "fish_id", "cls","x","y","w","h"]]
        return master_lbl_df.reset_index(drop=True)
    
    @staticmethod
    def pred_df_to_coco_json(pred_df_pth, save_json_pth):
        data = pd.read_csv(pred_df_pth, index_col=0)
        images = []
        categories = []
        annotations = []

        # Assign file and category ids
        data['fileid'] = data['filename'].astype('category').cat.codes
        data['categoryid'] = pd.Categorical(data['names'], ordered=True).codes
        data['annid'] = data.index

        # Build images list
        imagedf = data.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
        for row in imagedf.itertuples():
            image = {
                "height": row.im_h,
                "width": row.im_w,
                "image_id": str(row.filename).split(".")[0],
                "filename": row.filename
            }
            images.append(image)

        # Build categories list
        catdf = data.drop_duplicates(subset=['categoryid']).sort_values(by='categoryid')
        for row in catdf.itertuples():
            category = {
                "id": row.categoryid,
                "name": row.names,
                "supercategory": None
            }
            categories.append(category)

        # Build annotations list
        for row in data.itertuples():
            annotation = {
                "segmentation": [],
                "iscrowd": 0,
                "image_id": str(row.filename).split(".")[0],
                "bbox": [row.x * row.im_w, row.y * row.im_h, row.w * row.im_w, row.h * row.im_h],
                "category_id": row.categoryid,
                "id": row.annid,
                "score": row.conf
            }
            annotations.append(annotation)

        data_coco = {
            "images": images,
            "categories": categories,
            "annotations": annotations
        }
        with open(save_json_pth, "w") as f:
            json.dump(data_coco, f, indent=4)

    @staticmethod
    def make_master_lbl_df(lbl_path_lst):
        lbl_lst = []
        for row in lbl_path_lst:
            lbl_file= row
            lbl = pd.read_csv(lbl_file, sep=' ', names=["cls","x","y","w","h"], index_col=None)
            image_id = os.path.basename(lbl_file).split(".")[0]
            lbl['Filename'] = image_id
            lbl['id'] = lbl.index.astype(int)
            lbl_lst.append(lbl)
        master_lbl_df = pd.concat(lbl_lst, ignore_index=True)
        master_lbl_df['fish_id'] = master_lbl_df.Filename.astype(str) +"_"+ master_lbl_df.id.astype(str)
        master_lbl_df = master_lbl_df[["Filename", "id", "fish_id", "cls","x","y","w","h"]]
        return master_lbl_df.reset_index(drop=True)
    
    @staticmethod
    def master_lbl_df_to_coco_json(master_lbl_df, save_json_pth):
        data = master_lbl_df.copy()
        
        # COCO requires unique IDs for images and annotations, so we will use new IDs
        data['image_id'] = data['Filename'].astype('category').cat.codes
        data['category_id'] = pd.Categorical(data['cls'], ordered=True).codes
        # Create a unique annotation ID across the entire dataset
        data['annotation_id'] = data.index

        # Build images list
        imagedf = data.drop_duplicates(subset=['image_id']).sort_values(by='image_id')
        images = [
            {
                "id": row.image_id,
                "width": row.imw,
                "height": row.imh,
                "file_name": row.Filename + ".png"
            }
            for row in imagedf.itertuples()
        ]

        # Build categories list
        catdf = data.drop_duplicates(subset=['category_id']).sort_values(by='category_id')
        categories = [
            {
                "id": row.category_id,
                "name": str(row.cls),
                "supercategory": None
            }
            for row in catdf.itertuples()
        ]

        # Build annotations list
        annotations = []
        for row in data.itertuples():
            # Convert YOLO format (normalized center-x, center-y, w, h) to COCO format (top-left x, y, w, h)
            abs_x = (row.x - row.w / 2) * row.imw
            abs_y = (row.y - row.h / 2) * row.imh
            abs_w = row.w * row.imw
            abs_h = row.h * row.imh

            annotation = {
                "id": row.annotation_id,
                "image_id": row.image_id,
                "category_id": row.category_id,
                "bbox": [abs_x, abs_y, abs_w, abs_h],
                "area": abs_w * abs_h,  # Area of the bounding box
                "iscrowd": 0,
                "segmentation": []
            }
            annotations.append(annotation)

        data_coco = {
            "images": images,
            "categories": categories,
            "annotations": annotations
        }
        
        try:
            with open(save_json_pth, "w") as f:
                json.dump(data_coco, f, indent=4)
        except Exception as e:
            print(f"Error saving COCO JSON: {e}")
