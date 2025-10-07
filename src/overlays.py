import os
import cv2
import pandas as pd
import PIL
from PIL import ImageDraw, ImageFont
import PIL.Image as Image
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics.data.annotator import auto_annotate
import glob

class Overlays:
    @staticmethod
    def save_annot_imgs(img_pth, scores_df, save_path, conf_thresh, background = "image"):
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

    @staticmethod
    def save_annot_imgs2(img_pth, scores_df, lbl_df, save_path, conf_thresh, background = "image"):
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
    
    @staticmethod
    def save_annot_imgs_obb(img_pth, lbl_df, score_df, save_path, conf_thresh, background = "image"):
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

    @staticmethod
    def save_annot_imgs_pred_only(img_pth, pred_df, save_path, conf_thresh, background = "image"):
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

    @staticmethod
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
    
    @staticmethod
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
                cls, x, y, w, h = row[0], row[1]*im_w, row[2]*im_h, row[3]*im_w, row[4]*im_h
                # print(x,y,w,h)
                x1 = x - w/2
                y1 = y - h/2
                x2 = x + w/2
                y2 = y + h/2
                if int(cls) == 0:
                    draw.rectangle((x1,y1,x2,y2), outline="#FF5F1F", width=1)
                elif int(cls) == 1:
                    draw.rectangle((x1,y1,x2,y2), outline="#FF1493", width=1)
                else:
                    draw.rectangle((x1,y1,x2,y2), outline="#FFFF00", width=1)
                # draw.ellipse((x-s,y-s,x+s,y+s), fill=(200))
        except: print("label not processed")
        return img
    
    @staticmethod
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
    
    def plot_coco_boxes(image_path, json_path, target_size=(2048, 1500)):
        """
        Plots bounding boxes from a COCO JSON file onto a resized image.

        Args:
            image_path (str): The file path to the image.
            json_path (str): The file path to the COCO JSON annotation file.
            target_size (tuple): The desired display size as a tuple (width, height).
        """
        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image from path: {image_path}")
                return
            
            # Get original image dimensions
            original_h, original_w = image.shape[:2]

            # Resize the image for display
            resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            
            # Load the COCO JSON data
            with open(json_path, 'r') as f:
                coco_data = json.load(f)
                
            # Extract image filename from the path
            image_filename = os.path.basename(image_path)

            # Find the image ID from the JSON file
            image_id = -1
            for img in coco_data['images']:
                if img['file_name'] == image_filename:
                    image_id = img['id']
                    break
            
            if image_id == -1:
                print(f"Error: Image '{image_filename}' not found in JSON.")
                return

            # Create a plot
            fig, ax = plt.subplots(1, figsize=(12, 9))
            ax.imshow(resized_image)
            
            # Calculate scaling factors
            scale_x = target_size[0] / original_w
            scale_y = target_size[1] / original_h

            # Plot the bounding boxes for the corresponding image
            for ann in coco_data['annotations']:
                if ann['image_id'] == image_id:
                    bbox = ann['bbox']
                    
                    # Scale the bounding box coordinates
                    x_scaled = bbox[0] * scale_x
                    y_scaled = bbox[1] * scale_y
                    w_scaled = bbox[2] * scale_x
                    h_scaled = bbox[3] * scale_y
                    
                    rect = patches.Rectangle(
                        (x_scaled, y_scaled), w_scaled, h_scaled,
                        linewidth=0.5, edgecolor='r', facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Optional: Add category label
                    category_id = ann['category_id']
                    category_name = "unknown"
                    for cat in coco_data['categories']:
                        if cat['id'] == category_id:
                            category_name = cat['name']
                            break
                    plt.text(x_scaled, y_scaled - 5, category_name, color='red', fontsize=12)

            ax.axis('off')
            plt.show()

        except FileNotFoundError:
            print(f"Error: One of the files was not found. Please check paths:\nImage: {image_path}\nJSON: {json_path}")
        except json.JSONDecodeError:
            print(f"Error: The JSON file is not in a valid format.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    # Example usage with a specified target size
    # plot_coco_boxes('path/to/your/image.jpg', 'path/to/your/annotations.json', target_size=(2048, 1500))

    @staticmethod
    def plot_bbox(df, filename):
        img1_pred = df[df.filename == filename]
        img1_bbox = img1_pred[["x", "y", "w", "h"]].values.tolist()
        img1_fp = img1_pred.image_path.unique()[0]
        im = Image.open(img1_fp)
        # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        draw = ImageDraw.Draw(im)
        for bbox in img1_bbox:
            x, y, w, h = bbox
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
            draw.rectangle((x1,y1,x2,y2), outline=(0, 0, 0), width=4)
        im.show()

    @staticmethod
    def plot_xy_pts(df, filename):
        img1_pred = df[df.filename == filename]
        img1_pts = img1_pred[["x", "y"]].values.tolist()
        img1_fp = img1_pred.image_path.unique()[0]
        im = Image.open(img1_fp)
        # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        draw = ImageDraw.Draw(im)
        s=6
        for pt in img1_pts:
            x, y = pt
            draw.ellipse((x-s,y-s,x+s,y+s), fill=(200))
        im.show()
        
    @staticmethod
    def plot_segmentation(img_filepath, seg_txt_filepath):
        with open (seg_txt_filepath, "r") as f:
            segs = f.readlines()
        im = Image.open(img_filepath)
        # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        draw = ImageDraw.Draw(im)
        for seg in segs:
            s1c = int(seg[0])
            s1seg = seg[1:].split(" ")
            s1seg = list(filter(None, s1seg))
            s1seg = list(map(float, s1seg))
            segx, segy = np.array(s1seg[0::2])*4096, np.array(s1seg[1::2])*2176
            segxy = [(x, y) for x, y in zip(segx, segy)]
            draw.polygon(segxy, fill=None, outline="red", width=3)
        im.show()

    @staticmethod
    def sam_auto_annotate(df, filename, weights):
        img1_pred = df[df.filename == filename]
        img1_filepath = img1_pred.image_path.unique()[0]
        output_dir="./runs/sam_test"
        results = auto_annotate(data=img1_filepath, det_model=weights, sam_model="sam2_l.pt", output_dir=output_dir)
        res_pth = os.path.join(output_dir, filename.split(".")[0]+".txt")
        Overlays.plot_segmentation(img1_filepath, res_pth)

    @staticmethod
    def plot_label_overlays(image_list, label_list, output_dir, overwrite=False):
        ''' Function to plot overlays of images and labels '''
        df_l = pd.DataFrame({"label_path": label_list})
        df_l["Filename"] = df_l.label_path.apply(lambda x: os.path.basename(x).split(".")[0])
        df = pd.DataFrame({"image_path": image_list})
        df["Filename"] = df.image_path.apply(lambda x: os.path.basename(x).split(".")[0])
        df = df.merge(df_l, on="Filename")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for img_pth, lbl_pth in zip(df.image_path, df.label_path):
            out_image_path = os.path.join(output_dir, os.path.basename(img_pth))
            exists = os.path.exists(out_image_path)
            if overwrite:
                print(f"Processing {img_pth} and {lbl_pth}", end=" \r")
                img = Overlays.disp_lbl_bbox(img_pth, lbl_pth)
                img.save()
            else:
                if not exists:
                    print(f"Processing {img_pth} and {lbl_pth}", end=" \r")
                    img = Overlays.disp_lbl_bbox(img_pth, lbl_pth)
                    img.save(out_image_path)
                else:
                    print(f"Skipping {img_pth} as it already exists", end=" \r")