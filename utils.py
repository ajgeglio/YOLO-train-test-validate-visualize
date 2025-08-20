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
import zipfile
import fnmatch

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

    @staticmethod
    def seconds_to_hours_minutes(seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours} hours, {minutes} minutes"
    
    @staticmethod
    def list_files_in_zip(zip_path, pattern):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            matched_files = [f for f in file_list if fnmatch.fnmatch(f, pattern)]
        return matched_files

    @staticmethod
    def verify_images(images):
        """Verify images to ensure they are not corrupted."""
        for img_path in images:
            try:
                PIL.Image.open(img_path).verify()
            except:
                images.remove(img_path)
                print(f"Removing {img_path} from list because it is not compatible")
        return
    
    @staticmethod
    def initialize_logging(path, output_name, suppress_log):
        """Set up logging to a file if not suppressed."""
        if not suppress_log:
            nametime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file_path = os.path.join(path, f"{output_name}{nametime}.log")
            print("Saving a local log file in:", log_file_path)
            log_file = open(log_file_path, 'w')
            sys.stdout = log_file
        return sys.stdout
    
    @staticmethod
    def write_list_txt(filename, lst):
        with open(filename, "w") as f:
            f.writelines("%s\n" % l for l in lst)

    @staticmethod
    def make_imgs_nms_unique(img_pth_lst):
        # turn path strings into lists
        spl_pth_lst = list(map(lambda x: Path(x).parts, img_pth_lst))
        # using the last 
        new_img_nms_lst = list(map(lambda x: x[-3]+"_"+x[-2]+"_"+x[-1], spl_pth_lst))
        # remove spaces    
        new_img_nms_lst = list(map(lambda x: x.replace(" ", ""), new_img_nms_lst))  
        return new_img_nms_lst
    
    @staticmethod
    def nearest_index(array, value):
        # for getting approximately evently spaced indices from data randomly distributed
        return (np.abs(np.asarray(array) - value)).argmin()

    @staticmethod
    def evenly_spaced_indices(array, step=0.1):
        if step == 'AUC':
            return range(len(array))
        else:
            return [Utils.nearest_index(array, value) for value in np.arange(np.min(array), np.max(array), step)]

    @staticmethod
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

    @staticmethod
    def list_folders_w_pattern(pth, pat):
        folders = []
        for root, dirs, files in os.walk(pth):
            for name in dirs:
                folders.append(os.path.join(root,name))
        extra_folders = [re.findall(pat,folder) for folder in folders]
        folders = [sublist for list in extra_folders for sublist in list]
        return folders

    @staticmethod
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

    @staticmethod
    def list_files(filepath, filetype):
        paths = []
        for root, dirs, files in os.walk(filepath):
            for file in files:
                if file.lower().endswith(filetype.lower()):
                    paths.append(os.path.join(root, file))
        return(paths)

    @staticmethod
    def create_empty_txt_files(empty_id_list, ext=".txt"):
        for fil in empty_id_list:
            file = fil + f"{ext}"
            with open(file, 'w'):
                continue

    @staticmethod
    def make_move_df(orig_pth, new_pth, ext = ".txt"):
        descr = os.path.join(orig_pth,"*"+ext)
        files = glob.glob(descr)
        df = pd.DataFrame(files, columns=["original_path"])
        bn = lambda x: os.path.basename(x)
        df['original_filename'] = df.original_path.apply(bn)
        np = lambda x: os.path.join(new_pth,x)
        df['new_path'] = df.original_filename.apply(np)
        return df

    @staticmethod
    def move_files_lst(orig_pth, new_pth, ext = ".txt"):
        move_df = Utils.make_move_df(orig_pth, new_pth, ext = ".txt")
        i = 0
        k = 0
        l = len(move_df)
        for src, dst in zip(move_df.original_path, move_df.new_path):
            if not os.path.exists(dst):
                shutil.move(src,dst)
                k+=1
            i+=1
            print("file", i,"/",l,"new items found", k, end=" \r")

    @staticmethod
    def make_timestamp_folder():
        t = datetime.datetime.now()
        timestring = f"{t.year:02d}{t.month:02d}{t.day:02d}-{t.hour:02d}{t.minute:02d}{t.second:02d}"
        Ymmdd = timestring.split("-")[0]
        out_folder = f"2019-2023-{Ymmdd}"
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        print(out_folder)
        return out_folder, Ymmdd
    
    @staticmethod
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

    @staticmethod
    def copy_files_lst(pth_lst, dest, overwrite=True):
        if not os.path.exists(dest):
            os.makedirs(dest)
        n_files = len(pth_lst)
        for i, pth in enumerate(pth_lst):
            i += 1
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
    
    @staticmethod
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

    @staticmethod
    def get_shape_pil(self, fname):
        try:
            img=PIL.Image.open(fname)
        except: img = PIL.Image.new("RGB", (10,10))
        return (img.height, img.width)
    
    @staticmethod
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
    
    @staticmethod
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
    
    @staticmethod
    def return_n_objects_in_lbl(lbl_file):
        try:
            n_objects = pd.read_csv(lbl_file, delimiter=' ', header=None).shape[0]
        except: n_objects = 0
        return n_objects
    
    @staticmethod
    def count_n_objects_in_split(directory, split):
        lbl_lst_pths = glob.glob(os.path.join(directory,split,"labels","*.txt"))
        i = 0
        for lbl_file in lbl_lst_pths:
            num_objects = Utils.return_objects_in_lbl(lbl_file)
            i += num_objects
        print(f"num objects in {split} is {i}")
    
    @staticmethod
    def count_objects_in_id_list(directory, id_lst):
        lbl_lst = list(map(lambda x: str(x)+".txt", id_lst))
        lbl_lst_pths = list(map(lambda x: os.path.join(directory, x), lbl_lst))
        i = 0
        for lbl_file in lbl_lst_pths:
            num_objects = Utils.return_objects_in_lbl(lbl_file)
            i += num_objects
        print(f"num objects in list is {i}")

    @staticmethod
    def print_lbl_img_discrepancy(dir_img, dir_lbl):
        lbls = glob.glob(os.path.join(dir_lbl , "*"))
        imgs = glob.glob(os.path.join(dir_img, "*"))
        id = lambda x: str(os.path.basename(x)).split(".")[0]
        lbl_ids = list(map(id, lbls))
        img_ids = list(map(id, imgs))
        print("n lbls", len(lbl_ids))
        print("n imgs", len(img_ids))
        lbl_id_diff = set(lbl_ids).difference(set(img_ids))
        img_id_diff = set(img_ids).difference(set(lbl_ids))
        print(f"extra lbls {len(lbl_id_diff)}, extra imgs {len(img_id_diff)}")
        return lbl_id_diff, img_id_diff

    @staticmethod
    def sort_and_move_discrepancy(dir_has_extra, dir_is_correct, dst):
        has_extra = glob.glob(os.path.join(dir_has_extra , "*"))
        is_correct = glob.glob(os.path.join(dir_is_correct, "*.png"))
        id = lambda x: str(os.path.basename(x)).split(".")[0]
        lbnm = lambda x: str(x).split(".")[0] + ".txt"
        jpnm = lambda x: str(x) + ".jpg"
        has_extra_ids = list(map(id, has_extra))
        is_correct_ids = list(map(id, is_correct))
        print(len(has_extra))
        print(len(is_correct))
        id_diff = set(has_extra_ids).difference(set(is_correct_ids))
        to_mov = list(map(jpnm, id_diff))
        assert len(has_extra) - len(is_correct) == len(to_mov)

        i = 1
        n_files = len(to_mov)
        for fil in to_mov:
            file = os.path.join(dir_has_extra, fil)
            if os.path.exists(file):
                shutil.move(file, dst)
                print("sample", i,'/',n_files, end='  \r')
                i += 1
            else: print("no discrepency", end= "  \r")

    @staticmethod
    def count_objects(lbl_lst_csv):
        label_list = pd.read_csv(lbl_lst_csv)
        i = 0
        for label_file in label_list:
            df = pd.read_csv(label_file, delimiter=' ', header=None)
            num_objects = df.shape[0]
            i += num_objects
        print(f"num objects in {which_set} is {i}")

    @staticmethod
    def im_dims_ratio(img_list):
        n_2176 = 0
        n_3000 = 0
        for i in range(len(img_list)):
            img_file = img_list[i]
            imgp = PIL.Image.open(img_file)
            im_array = np.array(imgp)
            im_h = im_array.shape[0]
            if im_h == 3000:
                n_3000 += 1
            else:
                n_2176 += 1
        return n_2176, n_3000