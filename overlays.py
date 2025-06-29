import os
import cv2
import pandas as pd
import PIL
from PIL import ImageDraw, ImageFont
import PIL.Image as Image
import numpy as np


class Overlays:
    def save_annot_imgs(self, img_pth, scores_df, save_path, conf_thresh, background = "image"):
        img_name = os.path.basename(img_pth)
        img_id = img_name.split(".")[0]
        img_array = cv2.imread(img_pth)
        if background == "image":
            img_array = img_array[:, :, ::-1]
            img = PIL.Image.fromarray(img_array)
        else:
            im_h, im_w = img_array.shape[0], img_array.shape[1]
            img = PIL.Image.new("RGB", (im_w, im_h), color="white")
        draw = PIL.ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 16)
        score = scores_df[scores_df.Filename == img_id]
        score = score[score.conf >= conf_thresh]
        n_fish = score.shape[0]
        for index, row in score.iterrows():
            # print(index, row)
            xl, yl = row.loc["x_l"]*row.loc["imw"], row.loc["y_l"]*row.loc["imh"]
            wl, hl = row.loc["w_l"]*row.loc["imw"], row.loc["h_l"]*row.loc["imh"]
            xp, yp = row.loc["x_p"]*row.loc["imw"], row.loc["y_p"]*row.loc["imh"]
            wp, hp = row.loc["w_p"]*row.loc["imw"], row.loc["h_p"]*row.loc["imh"]
            conf = row.loc["conf"]
            detect_id = row.loc['detect_id'].split("_")[-1]
            ground_truth_id = row.loc['ground_truth_id'].split("_")[-1]
            pos = row.loc['tp']

            # label box
            x1 = xl - wl/2
            y1 = yl - hl/2
            x2 = xl + wl/2
            y2 = yl + hl/2
            # prediction box
            x12 = xp - wp/2
            y12 = yp - hp/2
            x22 = xp + wp/2
            y22 = yp + hp/2
            # draw label box and ground_truth_id in black
            draw.rectangle((x1,y1,x2,y2), outline="black", width=3)
            draw.text((x1+10, y2-25), text = ground_truth_id, fill="black", stroke=0, stroke_color=(200), font=font)
            text_ = detect_id+" @ "+f"{conf:0.2f}" # detection tag
            bbox1 = draw.textbbox((x12, y12), text = text_, font=font, anchor="lb") # top left
            bbox2 = draw.textbbox((x22, y22), text = text_, font=font, anchor='rt') # bottom right 
            if int(pos) == 1:
                # draw prediction box and text green
                draw.rectangle((x12,y12,x22,y22), outline="green", width=3)
                draw.rectangle(bbox1, fill="green") # top left
                draw.rectangle(bbox2, fill="green") # bottom right     
            else:
                # draw prediction box and text red
                draw.rectangle((x12,y12,x22,y22), outline="red", width=3)
                draw.rectangle(bbox1, fill="red") # top left
                draw.rectangle(bbox2, fill="red") # bottom right
            # draw prediction box and label with confidence and detection_id
            draw.text((x12, y12), text = text_, fill="white", stroke=8, stroke_color="white", font=font, anchor="lb")
            draw.text((x22, y22), text = text_, fill="white", stroke=8, stroke_color="white", font=font, anchor='rt')
        if n_fish > 0:
            img.save(os.path.join(save_path,f"{img_id}_a.jpg")) 
        else: print("no fish enumerated or dectected")

    def save_annot_imgs2(self, img_pth, scores_df, lbl_df, save_path, conf_thresh, background = "image"):
        img_name = os.path.basename(img_pth)
        img_id = img_name.split(".")[0]
        img_array = cv2.imread(img_pth)
        if background == "image":
            img_array = img_array[:, :, ::-1]
            im_h, im_w = img_array.shape[0], img_array.shape[1]
            img = PIL.Image.fromarray(img_array)
        else:
            im_h, im_w = img_array.shape[0], img_array.shape[1]
            img = PIL.Image.new("RGB", (im_w, im_h), color="white")
        draw = PIL.ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 16)
        score = scores_df[scores_df.Filename == img_id]
        score = score[score.conf >= conf_thresh]
        lbl = lbl_df[lbl_df.Filename == img_id]
        n_fish = lbl.shape[0]
        for index, row in lbl.iterrows():
            ground_truth_id = row.loc['ground_truth_id'].split("_")[-1]
            xl, yl = row.loc["x"]*im_w, row.loc["y"]*im_h
            wl, hl = row.loc["w"]*im_w, row.loc["h"]*im_h
            # label box
            x1 = xl - wl/2
            y1 = yl - hl/2
            x2 = xl + wl/2
            y2 = yl + hl/2
            # draw label box and ground_truth_id in black
            draw.rectangle((x1,y1,x2,y2), outline="black", width=3)
            draw.text((x1+10, y2-25), text = ground_truth_id, fill="black", stroke=0, stroke_color=(200), font=font)
        for index, row in score.iterrows():
            # print(index, row)
            xp, yp = row.loc["x_p"]*row.loc["imw"], row.loc["y_p"]*row.loc["imh"]
            wp, hp = row.loc["w_p"]*row.loc["imw"], row.loc["h_p"]*row.loc["imh"]
            conf = row.loc["conf"]
            detect_id = row.loc['detect_id'].split("_")[-1]
            pos = row.loc['tp']

            # prediction box
            x12 = xp - wp/2
            y12 = yp - hp/2
            x22 = xp + wp/2
            y22 = yp + hp/2
            text_ = detect_id+" @ "+f"{conf:0.2f}" # detection tag
            bbox1 = draw.textbbox((x12, y12), text = text_, font=font, anchor="lb") # top left
            bbox2 = draw.textbbox((x22, y22), text = text_, font=font, anchor='rt') # bottom right 
            if int(pos) == 1:
                # draw prediction box and text green
                draw.rectangle((x12,y12,x22,y22), outline="green", width=3)
                draw.rectangle(bbox1, fill="green") # top left
                draw.rectangle(bbox2, fill="green") # bottom right     
            else:
                # draw prediction box and text red
                draw.rectangle((x12,y12,x22,y22), outline="red", width=3)
                draw.rectangle(bbox1, fill="red") # top left
                draw.rectangle(bbox2, fill="red") # bottom right
            # draw prediction box and label with confidence and detection_id
            draw.text((x12, y12), text = text_, fill="white", stroke=8, stroke_color="white", font=font, anchor="lb")
            draw.text((x22, y22), text = text_, fill="white", stroke=8, stroke_color="white", font=font, anchor='rt')
        if n_fish > 0:
            img.save(os.path.join(save_path,f"{img_id}_a.jpg")) 
        else: print("no fish enumerated or dectected")
    
    def save_annot_imgs_hybrid(self, img_pth, hybrid_df, pred_df, save_path, conf_thresh, background = "image"):
        def convert_bbox(row):
            x, y, w, h = row.loc['bbox']
            # xmax, xmin = x + w/2, x - w/2
            xmax, xmin = x + w, x
            ymax, ymin = y + h, y
            return [xmin, ymin, xmax, ymax]
        img_name = os.path.basename(img_pth)
        img_id = img_name.split(".")[0]
        img_array = cv2.imread(img_pth)
        if background == "image":
            img_array = img_array[:, :, ::-1]
            img = PIL.Image.fromarray(img_array)
        else:
            im_h, im_w = img_array.shape[0], img_array.shape[1]
            img = PIL.Image.new("RGB", (im_w, im_h), color="white")
        draw = PIL.ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 16)
        lbls = hybrid_df[hybrid_df.image_id == img_id]
        lbls = lbls[lbls.score>=1]
        pred = pred_df[pred_df.image_id == img_id]
        pred = pred[pred.score >= conf_thresh]
        n_fish = lbls.shape[0]
        for idx, row in lbls.iterrows():
            xmin, ymin, xmax, ymax = convert_bbox(row) # label box
            ground_truth_id = str(idx)
            # draw label box and ground_truth_id in black
            draw.rectangle((xmin,ymin,xmax,ymax), outline="black", width=3)
            draw.text((xmin+10, ymax-25), text = ground_truth_id, fill="black", stroke=0, stroke_color=(200), font=font)
        for idx, row in pred.iterrows():
            xmin, ymin, xmax, ymax = convert_bbox(row) # prediction box
            conf = row.loc["score"]
            detect_id = "dt" + str(idx)
            # draw prediction box and text red
            draw.rectangle((xmin,ymin,xmax,ymax), outline="red", width=2)
            text_ = detect_id+" @ "+f"{conf:0.2f}" # detection tag
            bbox1 = draw.textbbox((xmin, ymin), text = text_, font=font, anchor="lb") # top left
            bbox2 = draw.textbbox((xmax, ymax), text = text_, font=font, anchor='rt') # bottom right 
            draw.rectangle(bbox1, fill="red") # top left
            draw.rectangle(bbox2, fill="red") # bottom right
            # draw prediction box and label with confidence and detection_id
            draw.text((xmin, ymin), text = text_, fill="white", stroke=8, stroke_color="white", font=font, anchor="lb")
            draw.text((xmax, ymax), text = text_, fill="white", stroke=8, stroke_color="white", font=font, anchor='rt')
        if n_fish > 0:
            img.save(os.path.join(save_path,f"{img_id}_a.jpg")) 

    def save_annot_imgs_obb(self, img_pth, lbl_df, score_df, save_path, conf_thresh, background = "image"):
        img_name = os.path.basename(img_pth)
        img_id = img_name.split(".")[0]
        img_array = cv2.imread(img_pth)
        im_h, im_w = img_array.shape[0], img_array.shape[1]
        if background == "image":
            img_array = img_array[:, :, ::-1]
            img = PIL.Image.fromarray(img_array)
        else:
            im_h, im_w = img_array.shape[0], img_array.shape[1]
            img = PIL.Image.new("RGB", (im_w, im_h), color="white")
        draw = PIL.ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 16)
        score = score_df[score_df.image_id == img_id]
        score = score[score.conf >= conf_thresh]
        lbl = lbl_df[lbl_df.image_id == img_id]
        n_fish = lbl.shape[0]
        for index, row in lbl.iterrows():
            xs = row['x1'], row['x2'], row['x3'], row['x4']
            ys = row['y1'], row['y2'], row['y3'], row['y4']
            # imw, imh = row['im_w'], row['im_h']
            x1, x2, x3, x4 = [x*im_w for x in xs]
            y1, y2, y3, y4 = [y*im_h for y in ys]
            bbox1 = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            ground_truth_id = row.loc['ground_truth_id'].split("_")[-1]
            # label box
            # draw label box and ground_truth_id in black
            # draw.polygon(bbox1, outline=(0, 0, 0), width=3)
            draw.polygon(bbox1, outline="black", width=3)
            draw.text((np.average([x1, x2, x3, x4]), np.average([y1, y2, y3, y4])), text = ground_truth_id, fill="black", stroke=0, stroke_color=(200), font=font)
            # prediction box
        for index, row in score.iterrows():    
            xsp = row['x1_p'], row['x2_p'], row['x3_p'], row['x4_p']
            ysp = row['y1_p'], row['y2_p'], row['y3_p'], row['y4_p']
            x1p, x2p, x3p, x4p = [x*im_w for x in xsp]
            y1p, y2p, y3p, y4p = [y*im_h for y in ysp]
            bbox2 = [(x1p, y1p), (x2p, y2p), (x3p, y3p), (x4p, y4p)]
            detect_id = row.loc['detect_id'].split("_")[-1]
            conf = row.loc["conf"]
            text_ = detect_id+" @ "+f"{conf:0.2f}" # detection tag
            bbox1t = draw.textbbox((x1p, y1p), text = text_, font=font, anchor="lb") # top left
            bbox2t = draw.textbbox((x3p, y3p), text = text_, font=font, anchor='rt') # bottom right 
            pos = row.loc['tp']
            if int(pos) == 1:
                # draw prediction box and text green
                draw.polygon(bbox2, outline="green", width=3)
                draw.rectangle(bbox1t, fill="green") # top left
                draw.rectangle(bbox2t, fill="green") # bottom right     
            else:
                # draw prediction box and text red
                draw.polygon(bbox2, outline="red", width=3)
                draw.rectangle(bbox1t, fill="red") # top left
                draw.rectangle(bbox2t, fill="red") # bottom right
            # draw prediction box and label with confidence and detection_id
            draw.text((x1p, y1p), text = text_, fill="white", stroke=8, stroke_color="white", font=font, anchor="lb")
            draw.text((x3p, y3p), text = text_, fill="white", stroke=8, stroke_color="white", font=font, anchor='rt')
        if n_fish > 0:
            img.save(os.path.join(save_path,f"{img_id}_a.jpg")) 

    def save_annot_imgs_pred_only(self, img_pth, pred_df, save_path, conf_thresh, background = "image"):
        img_name = os.path.basename(img_pth)
        img_id = img_name.split(".")[0]
        filename = os.path.join(save_path, f"{img_id}_i.jpg")
        pred = pred_df[pred_df.Filename == img_id]
        pred = pred[pred.conf >= conf_thresh]
        n_fish = pred.shape[0]
        if n_fish == 0:
            return
        if os.path.exists(filename):
            return
        else:
            img_array = cv2.imread(img_pth)
            if background == "image":
                img_array = img_array[:, :, ::-1]
                img = PIL.Image.fromarray(img_array)
            else:
                im_h, im_w = img_array.shape[0], img_array.shape[1]
                img = PIL.Image.new("RGB", (im_w, im_h), color="white")
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("arial.ttf", size=20)
            for index, row in pred.iterrows():
                xp, yp = row.loc["x"] * row.loc["imw"], row.loc["y"] * row.loc["imh"]
                wp, hp = row.loc["w"] * row.loc["imw"], row.loc["h"] * row.loc["imh"]
                conf = row.loc["conf"]
                detect_id = row.loc['detect_id'].split("_")[-1]
                x12 = xp - wp / 2
                y12 = yp - hp / 2
                x22 = xp + wp / 2
                y22 = yp + hp / 2
                text_ = detect_id + " @ " + f"{conf:0.2f}"
                bbox1 = draw.textbbox((x12, y12), text=text_, font=font, anchor="lb")
                bbox2 = draw.textbbox((x22, y22), text=text_, font=font, anchor='rt')
                draw.rectangle((x12, y12, x22, y22), outline="red", width=3)
                draw.rectangle(bbox1, fill="red")
                draw.rectangle(bbox2, fill="red")
                draw.text((x12, y12), text=text_, fill="white", stroke=8, stroke_color="white", font=font, anchor="lb")
                draw.text((x22, y22), text=text_, fill="white", stroke=8, stroke_color="white", font=font, anchor='rt')
                img.save(filename)

    def disp_bbox_mask(mask_array, df):
        # mask_array = cv2.imread(msk_file)
        msk_img = PIL.Image.fromarray(mask_array)
        # im_h, im_w = mask_array.shape[0], mask_array.shape[1]
        draw = PIL.ImageDraw.Draw(msk_img)
        s=6
        for index, row in df.iterrows():
            x, y, w, h = row[1], row[2], row[3], row[4]
            # print(x,y,w,h)
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
            draw.rectangle((x1,y1,x2,y2), outline=(200), width=1)
            draw.ellipse((x-s,y-s,x+s,y+s), fill=(200))
        return msk_img
    def disp_lbl_bbox(img_path, lbl_path):
        assert os.path.basename(img_path).split(".")[0] ==  os.path.basename(lbl_path).split(".")[0]
        img_array = cv2.imread(img_path)[:, :, ::-1]
        img = PIL.Image.fromarray(img_array)
        im_h, im_w = img_array.shape[0], img_array.shape[1]
        draw = PIL.ImageDraw.Draw(img)
        s=6
        try:
            df = pd.read_csv(lbl_path, delimiter=' ', header=None)
            for index, row in df.iterrows():
                x, y, w, h = row[1]*im_w, row[2]*im_h, row[3]*im_w, row[4]*im_h
                # print(x,y,w,h)
                x1 = x - w/2
                y1 = y - h/2
                x2 = x + w/2
                y2 = y + h/2
                draw.rectangle((x1,y1,x2,y2), outline="black", width=1)
                # draw.ellipse((x-s,y-s,x+s,y+s), fill=(200))
        except: print("label not processed")
        return img
    
    def disp_bbox_only(img_path, lbl_path):
        assert os.path.basename(img_path).split(".")[0] ==  os.path.basename(lbl_path).split(".")[0]
        img_array = cv2.imread(img_path)
        im_h, im_w = img_array.shape[0], img_array.shape[1]
        blank = Image.new("RGB", (im_w, im_h), color="white")
        draw = PIL.ImageDraw.Draw(blank)
        s=6
        try:
            df = pd.read_csv(lbl_path, delimiter=' ', header=None)
            # df = lbl
            for index, row in df.iterrows():
                x, y, w, h = row[1]*im_w, row[2]*im_h, row[3]*im_w, row[4]*im_h
                # print(x,y,w,h)
                x1 = x - w/2
                y1 = y - h/2
                x2 = x + w/2
                y2 = y + h/2
                draw.rectangle((x1,y1,x2,y2), outline="black", width=1)
                # draw.ellipse((x-s,y-s,x+s,y+s), fill=(200))
        except: pass
        return blank
    
    def disp_merr_bbox(img_path, mlbl_path):
        assert os.path.basename(img_path).split(".")[0] ==  os.path.basename(mlbl_path).split(".")[0]
        img_array = cv2.imread(img_path)[:, :, ::-1]
        img = PIL.Image.fromarray(img_array)
        im_h, im_w = img_array.shape[0], img_array.shape[1]
        draw = PIL.ImageDraw.Draw(img)
        s=6
        try:
            df = pd.read_csv(mlbl_path, delimiter=' ', header=None)
            for i, r in df.iterrows():
                x1, x2, x3, x4 = r[1]*im_w, r[3]*im_w, r[5]*im_w, r[7]*im_w
                y1, y2, y3, y4 = r[2]*im_h, r[4]*im_h, r[6]*im_h, r[8]*im_h
                draw.polygon([x1, y1, x2, y2, x3, y3, x4, y4], outline="red", width=1)
        except: pass
        return img