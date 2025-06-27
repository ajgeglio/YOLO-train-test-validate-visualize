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