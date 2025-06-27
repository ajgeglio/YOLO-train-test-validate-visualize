import os
import re
import glob
import shutil
import datetime
import numpy as np
import pandas as pd
import PIL.Image
from pathlib import Path
import sys
import json

class ReturnTime:
    def __init__(self):
        pass

    @staticmethod
    def get_time_obj(time_s):
        if pd.notnull(time_s):
            return datetime.datetime.fromtimestamp(time_s)
        return np.nan

    @classmethod
    def get_Y(cls, time_s):
        dt = cls.get_time_obj(time_s)
        return dt.strftime('%Y') if isinstance(dt, datetime.datetime) else np.nan

    @classmethod
    def get_m(cls, time_s):
        dt = cls.get_time_obj(time_s)
        return dt.strftime('%m') if isinstance(dt, datetime.datetime) else np.nan

    @classmethod
    def get_d(cls, time_s):
        dt = cls.get_time_obj(time_s)
        return dt.strftime('%d') if isinstance(dt, datetime.datetime) else np.nan

    @classmethod
    def get_t(cls, time_s):
        dt = cls.get_time_obj(time_s)
        return dt.strftime('%H:%M:%S') if isinstance(dt, datetime.datetime) else np.nan

class Utils:
    def __init__(self) -> None:
        pass

    def seconds_to_hours_minutes(seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours} hours, {minutes} minutes"
    
    def verify_images(images):
        """Verify images to ensure they are not corrupted."""
        for img_path in images:
            try:
                PIL.Image.open(img_path).verify()
            except:
                images.remove(img_path)
                print(f"Removing {img_path} from list because it is not compatible")
        return
    
    def initialize_logging(path, output_name, suppress_log):
        """Set up logging to a file if not suppressed."""
        if not suppress_log:
            nametime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file_path = os.path.join(path, f"{output_name}{nametime}.log")
            print("Saving a local log file in:", log_file_path)
            log_file = open(log_file_path, 'w')
            sys.stdout = log_file
        return sys.stdout
    
    def write_list_txt(filename, lst):
        with open(filename, "w") as f:
            f.writelines("%s\n" % l for l in lst)

    def make_imgs_nms_unique(img_pth_lst):
        # turn path strings into lists
        spl_pth_lst = list(map(lambda x: Path(x).parts, img_pth_lst))
        # using the last 
        new_img_nms_lst = list(map(lambda x: x[-3]+"_"+x[-2]+"_"+x[-1], spl_pth_lst))
        # remove spaces    
        new_img_nms_lst = list(map(lambda x: x.replace(" ", ""), new_img_nms_lst))  
        return new_img_nms_lst
    
    def nearest_index(self, array, value):
        # for getting approximately evently spaced indices from data randomly distributed
        return (np.abs(np.asarray(array) - value)).argmin()

    def evenly_spaced_indices(self, array, step=0.1):
        if step == 'AUC':
            return range(len(array))
        else:
            return [self.nearest_index(array, value) for value in np.arange(np.min(array), np.max(array), step)]
        
    def list_files_exclude_pattern(filepath, filetype, pat):
        paths = []
        for root, dirs, files in os.walk(filepath):
            if re.search(pat, root):
                pass
            else:
                for file in files:
                    if file.lower().endswith(filetype.lower()):
                        paths.append(os.path.join(root, file))
        return(paths)

    def list_folders_w_pattern(pth, pat):
        folders = []
        for root, dirs, files in os.walk(pth):
            for name in dirs:
                folders.append(os.path.join(root,name))
        extra_folders = [re.findall(pat,folder) for folder in folders]
        folders = [sublist for list in extra_folders for sublist in list]
        return folders

    def list_collects(filepath):
        paths = []
        for root, dirs, files in os.walk(filepath):
            for dir in dirs:
                paths.append(os.path.join(root, dir))
                pat1 = '([0-9]{8}_[0-9]{3}_[a-z,A-Z]+[0-9][0-9][0-9][0-9]_[a-z,A-Z]+[0-2])'
                # pat2 = '.*d+_\d+_\w+_\w{4}\Z'
        collects = [re.findall(pat1, i) for i in paths]
        collects = list(set([item for sublist in collects for item in sublist]))
        return(collects)

    def list_files(filepath, filetype):
        paths = []
        for root, dirs, files in os.walk(filepath):
            for file in files:
                if file.lower().endswith(filetype.lower()):
                    paths.append(os.path.join(root, file))
        return(paths)

    def create_empty_txt_files(empty_id_list, ext=".txt"):
        for fil in empty_id_list:
            file = fil + f"{ext}"
            with open(file, 'w'):
                continue

    def make_move_df(self, orig_pth, new_pth, ext = ".txt"):
        descr = os.path.join(orig_pth,"*"+ext)
        files = glob.glob(descr)
        df = pd.DataFrame(files, columns=["original_path"])
        bn = lambda x: os.path.basename(x)
        df['original_filename'] = df.original_path.apply(bn)
        np = lambda x: os.path.join(new_pth,x)
        df['new_path'] = df.original_filename.apply(np)
        return df

    def move_files_lst(self, orig_pth, new_pth, ext = ".txt"):
        move_df = self.make_move_df(orig_pth, new_pth, ext = ".txt")
        i = 0
        k = 0
        l = len(move_df)
        for src, dst in zip(move_df.original_path, move_df.new_path):
            if not os.path.exists(dst):
                shutil.copy(src,dst)
                k+=1
            i+=1
            print("file", i,"/",l,"new items found", k, end=" \r")

    def make_timestamp_folder():
        t = datetime.datetime.now()
        timestring = f"{t.year:02d}{t.month:02d}{t.day:02d}-{t.hour:02d}{t.minute:02d}{t.second:02d}"
        Ymmdd = timestring.split("-")[0]
        out_folder = f"2019-2023-{Ymmdd}"
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        print(out_folder)
        return out_folder, Ymmdd
    
    def create_collect_id_df(imgs_glob):
        ''' create a dataframe of collect ids based on a list of images'''
        df = pd.DataFrame()
        df['image_path'] = imgs_glob
        df['filename'] = df.image_path.apply(lambda x: os.path.basename(x))
        folder = df.image_path.apply(lambda x: os.path.basename(os.path.dirname(os.path.dirname(x))))
        pattern = r"([0-9]{8}_[0-9]{3}_[a-z,A-Z]{4}[0-9]{4}_[a-z,A-Z]{3}[0-2]{1})"
        collect_id_match = folder.apply(lambda x: re.search(pattern, x))
        collect_ids = [match.group() for match in collect_id_match if match != None]
        col_idx = collect_id_match[collect_id_match.values!=None].index
        df = df.loc[col_idx]
        df['collect_id'] = collect_ids
        return df

    def copy_files_lst(pth_lst, dest, overwrite=True):
        if not os.path.exists(dest):
            os.makedirs(dest)
        n_files = len(pth_lst)
        for i, pth in enumerate(pth_lst):
            if pth == pth:
                if overwrite == True:
                    nm = os.path.basename(pth)
                    new_pth = os.path.join(dest, nm)
                    print("sample", i,'/',n_files, end='  \r')
                    shutil.copy2(pth, new_pth)
                elif overwrite == False:
                    org_sz = os.path.getsize(pth)
                    nm = os.path.basename(pth)
                    new_pth = os.path.join(dest, nm)
                    if not os.path.exists(new_pth):
                        print("sample", i,'/',n_files, end='  \r')
                        shutil.copy2(pth, new_pth)
                    else:
                        exist_size = os.path.getsize(new_pth)
                        if org_sz != exist_size:
                            print("sample", i,'/',n_files, end='  \r')
                            shutil.copy2(pth, new_pth)
                        else: continue #the exact file name and size already exists
            else: continue
    
    def copy_and_replace_directory(src, dst):
        n_dir = len(os.listdir(dst))
        print("dst dir has", n_dir, "files")
        src_files = glob.glob(os.path.join(src , "*"))
        i = 1
        n_files = len(src_files)
        print("copy src has", n_files, "files")
        assert n_dir == n_files, "source and destination n files do not match, first solve discrepency"
        for fil in src_files:
            shutil.copy2(fil, dst)
            print("sample", i,'/',n_files, end='  \r')
            i += 1

    def get_shape_pil(self, fname):
        try:
            img=PIL.Image.open(fname)
        except: img = PIL.Image.new("RGB", (10,10))
        return (img.height, img.width)
    
    def make_img_df_folder(self, img_folder, ext=".png"):
    # use to check a list of images and get dates times and their dimensions(not good for very large lists)
        img_pths = glob.glob(os.path.join(img_folder,"*"+ext))
        tim = lambda x: os.path.basename(x)
        tid = lambda x: str(x).split(".")[0]
        tet = lambda x: float(str(x).split("_")[1]+"."+str(x).split("_")[2])
        tdt = lambda x: datetime.datetime.fromtimestamp(x)
        ims = lambda x: self.get_shape_pil(x)
        df = pd.DataFrame(img_pths, columns=["image_path"])
        df["image_dim"] = df.image_path.apply(ims)
        df["image_id"] = df.image_path.apply(tim).apply(tid)
        df["time_s"] = df.image_id.apply(tet)
        df["datetime"] = df.time_s.apply(tdt)
        return df
    
    def make_img_df_list(self, img_pths):
        tim = lambda x: os.path.basename(x)
        tid = lambda x: str(x).split(".")[0]
        tet = lambda x: float(str(x).split("_")[1]+"."+str(x).split("_")[2])
        tdt = lambda x: datetime.datetime.fromtimestamp(x)
        ims = lambda x: self.get_shape_pil(x)
        df = pd.DataFrame(img_pths, columns=["image_path"])
        df["image_dim"] = df.image_path.apply(ims)
        df["image_id"] = df.image_path.apply(tim).apply(tid)
        df["time_s"] = df.image_id.apply(tet)
        df["datetime"] = df.time_s.apply(tdt)
        return df
    
    def return_objects_in_lbl(self, lbl_file):
        try:
            n_fish = pd.read_csv(lbl_file, delimiter=' ', header=None).shape[0]
        except: n_fish = 0
        return n_fish
    
    def count_objects_in_split(self, directory, split):
        lbl_lst_pths = glob.glob(os.path.join(directory,split,"labels","*.txt"))
        i = 0
        for lbl_file in lbl_lst_pths:
            num_objects = self.return_objects_in_lbl(lbl_file)
            i += num_objects
        print(f"num objects in {split} is {i}")
    
    def count_objects_in_id_list(self, directory, id_lst):
        lbl_lst = list(map(lambda x: str(x)+".txt", id_lst))
        lbl_lst_pths = list(map(lambda x: os.path.join(directory, x), lbl_lst))
        i = 0
        for lbl_file in lbl_lst_pths:
            num_objects = self.return_objects_in_lbl(lbl_file)
            i += num_objects
        print(f"num objects in list is {i}")

    def make_master_lbl_df(lbl_path_lst_yr):
        lbl_lst = []
        for idx, row in lbl_path_lst_yr.iterrows():
            lbl_file, yr = row
            lbl = pd.read_csv(lbl_file, sep=' ', names=["cls","x","y","w","h"], index_col=None)
            image_id = os.path.basename(lbl_file).split(".")[0]
            lbl['Filename'] = image_id
            lbl['id'] = lbl.index.astype(int)
            lbl['year'] = yr
            lbl_lst.append(lbl)
        master_lbl_df = pd.concat(lbl_lst, ignore_index=True)
        master_lbl_df['fish_id'] = master_lbl_df.image_id.astype(str) +"_"+ master_lbl_df.id.astype(str)
        master_lbl_df = master_lbl_df[["Filename", "id", "fish_id", "cls","x","y","w","h", "year"]]
        return master_lbl_df.reset_index(drop=True)
        
    def make_master_lbl_df_gopro(lbl_path_lst):
        lbl_lst = []
        for lbl_file in lbl_path_lst:
            lbl = pd.read_csv(lbl_file, sep=' ', names=["cls","x","y","w","h"], index_col=None)
            image_id = os.path.basename(lbl_file).split(".")[0]
            lbl['image_id'] = image_id
            lbl['id'] = lbl.index.astype(int)
            lbl_lst.append(lbl)
        master_lbl_df = pd.concat(lbl_lst, ignore_index=True)
        master_lbl_df['fish_id'] = master_lbl_df.image_id.astype(str) +"_"+ master_lbl_df.id.astype(str)
        master_lbl_df = master_lbl_df[["image_id", "id", "fish_id", "cls","x","y","w","h"]]
        return master_lbl_df.reset_index(drop=True)
    
    def pred_df_to_coco_json(pred_df_pth, save_json_pth):
        data = pd.read_csv(pred_df_pth, index_col=0)
        images = []
        categories = []
        annotations = []

        category = {}
        category["supercategory"] = None
        category["id"] = None
        category["name"] = None
        categories.append(category)
        
        data['fileid'] = data['filename'].astype('category').cat.codes
        data['categoryid']= pd.Categorical(data['names'],ordered= True).codes
        data['annid'] = data.index

        def image(row):
            image = {}
            image["height"] = row.im_h
            image["width"] = row.im_w
            image["image_id"] = str(row.filename).split(".")[0]
            image["filename"] = row.filename
            return image

        def category(row):
            category = {}
            # category["supercategory"] = row.supercategory
            category["id"] = row.categoryid
            category["name"] = row.names
            return category

        def annotation(row):
            annotation = {}
            # area = (row.xmax -row.xmin)*(row.ymax - row.ymin)
            annotation["segmentation"] = []
            annotation["iscrowd"] = 0
            # annotation["area"] = area
            annotation["image_id"] = str(row.filename).split(".")[0]
            annotation["bbox"] = [row.x*row.im_w, row.y*row.im_h, row.w*row.im_w, row.h*row.im_h]
            annotation["category_id"] = row.categoryid
            annotation["id"] = row.annid
            annotation["score"] = row.conf
            return annotation

        for row in data.itertuples():
            annotations.append(annotation(row))

        imagedf = data.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
        for row in imagedf.itertuples():
            images.append(image(row))

        catdf = data.drop_duplicates(subset=['categoryid']).sort_values(by='categoryid')
        for row in catdf.itertuples():
            categories.append(category(row))

        data_coco = {}
        data_coco["images"] = images
        data_coco["categories"] = categories
        data_coco["annotations"] = annotations
        json.dump(data_coco, open(save_json_pth, "w"), indent=4)