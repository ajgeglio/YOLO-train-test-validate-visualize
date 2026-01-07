import os
import re
import glob
import shutil
import datetime
import numpy as np
import pandas as pd
import PIL.Image
from pathlib import Path
from tqdm import tqdm
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

# --- Logger class to tee output to both file and terminal ---
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

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
    def verify_images_corrected(images):
        """
        Verifies images and returns a new list containing only the non-corrupt paths.
        
        Args:
            images (list): A list of image file paths (strings).
            
        Returns:
            list: A new list of valid image file paths.
        """
        valid_images = []
        
        # Use enumerate to track position and len to print progress
        total_images = len(images)
        print(f"Starting image verification for {total_images} images...")

        for i, img_path in enumerate(images):
            try:
                # 1. Open the image file
                img = PIL.Image.open(img_path)
                # 2. Verify the image integrity without fully loading it
                img.verify()
                # 3. Re-open and load the image content fully to catch deferred errors
                img = PIL.Image.open(img_path)
                img.load()
                
                # If both steps succeed, the path is added to the valid list
                valid_images.append(img_path)

            except Exception as e:
                # Print the path of the corrupt file and the error
                print(f"❌ Removing corrupt image: {img_path}. Error: {e}")
            
            # Optional: Print progress every 100 images
            if (i + 1) % 100 == 0 or (i + 1) == total_images:
                print(f"Processed {i + 1}/{total_images} images.")

        corrupt_count = total_images - len(valid_images)
        print(f"\nVerification complete. Removed {corrupt_count} corrupt images.")
        corrupt_images = [img for img in images if img not in valid_images]
        return valid_images, corrupt_images
    
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
    def write_list_txt(lst, filename):
        with open(filename, "w", encoding='utf-8') as f: 
            f.writelines("%s\n" % l for l in lst)

    @staticmethod
    def read_list_txt(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

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
                    print("copy file", i,'/',n_files, end='  \r')
                    shutil.copy2(pth, new_pth)
                elif overwrite == False:
                    org_sz = os.path.getsize(pth)
                    nm = os.path.basename(pth)
                    new_pth = os.path.join(dest, nm)
                    if not os.path.exists(new_pth):
                        print("copy file", i,'/',n_files, end='  \r')
                        shutil.copy2(pth, new_pth)
                    else:
                        exist_size = os.path.getsize(new_pth)
                        if org_sz != exist_size:
                            print("copy file", i,'/',n_files, end='  \r')
                            shutil.copy2(pth, new_pth)
                        else: continue #the exact file name and size already exists
            else: continue
    
    @staticmethod
    def MOVE_files_lst(pth_lst, dest, overwrite=True):
        if not os.path.exists(dest):
            os.makedirs(dest)
        n_files = len(pth_lst)
        for i, pth in enumerate(pth_lst):
            i += 1
            if os.path.exists(pth):
                if overwrite == True:
                    nm = os.path.basename(pth)
                    new_pth = os.path.join(dest, nm)
                    print("sample", i,'/',n_files, end='  \r')
                    shutil.move(pth, new_pth)
                elif overwrite == False:
                    org_sz = os.path.getsize(pth)
                    nm = os.path.basename(pth)
                    new_pth = os.path.join(dest, nm)
                    if not os.path.exists(new_pth):
                        print("sample", i,'/',n_files, end='  \r')
                        shutil.move(pth, new_pth)
                    else:
                        exist_size = os.path.getsize(new_pth)
                        if org_sz != exist_size:
                            print("sample", i,'/',n_files, end='  \r')
                            shutil.move(pth, new_pth)
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
    def get_shape_pil(fname):
        try:
            img=PIL.Image.open(fname)
        except: img = PIL.Image.new("RGB", (10,10))
        return (img.height, img.width)
    
    @staticmethod
    def read_YOLO_lbl(lbl_file):
        lbl = pd.read_csv(lbl_file, sep=' ', names=["cls","x","y","w","h"], index_col=None)
        return lbl
    
    @staticmethod
    def save_YOLO_lbl(df, lbl_file):
        df.to_csv(lbl_file, header=None, sep=' ', index=False)

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
        print(f"num objects in {lbl_lst_csv} is {i}")

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
    
    @staticmethod
    def darwin_to_YOLO(filepath):
        ''' Convert Darwin JSON annotation to YOLO format text file'''
        dirname = os.path.dirname(filepath)
        filename = os.path.basename(filepath).split(".")[0]
        print(f"Converting {filename} from Darwin to YOLO format", end=" \r")
        # Load the JSON file
        with open(filepath, "r") as f:
            data = json.load(f)

        # Get image dimensions
        slot = data["item"]["slots"][0]
        img_width = slot["width"]
        img_height = slot["height"]
        name_dict = {"fish":2, "0":0, "1":1, "2":2}  # Example mapping of class names to IDs
        # Convert annotations to YOLO format
        yolo_lines = []
        for ann in data["annotations"]:
            name = ann["slot_names"][0]
            name = name_dict[name] if name in name_dict else 0
            bbox = ann["bounding_box"]
            w = bbox["w"] / img_width
            h = bbox["h"]/ img_height
            x_center = bbox["x"] / img_width + w/2 
            y_center = bbox["y"] / img_height + h/2
            yolo_lines.append(f"{name} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # Save to YOLO-style text file
        with open(f"{os.path.join(dirname,filename)}.txt", "w") as f:
            f.write("\n".join(yolo_lines))

    @staticmethod
    def match_images_to_labels(image_paths, label_paths):
        """
        Match image paths to label paths by basename (without extension).
        
        Args:
            image_paths (list): List of full image file paths.
            label_paths (list): List of full label file paths.

        Returns:
            matched_images (list): Image paths that have a corresponding label.
            matched_labels (list): Corresponding label paths.
        """
        print("Matching images to labels by basename...")
        # Create a set of label basenames (without extension)
        label_basenames = {os.path.splitext(os.path.basename(lbl))[0] for lbl in label_paths}

        matched_images = []
        matched_labels = []

        for img_path in image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            if base_name in label_basenames:
                matched_images.append(img_path)
                # Find the matching label path
                label_path = next(lbl for lbl in label_paths if os.path.splitext(os.path.basename(lbl))[0] == base_name)
                matched_labels.append(label_path)

        return matched_images, matched_labels
    
   
    @staticmethod
    def export_paths(image_paths, label_paths, dataset_folder, subfolder):
        ''' Helper function to export image and label paths ensuring they are matched by basename. '''
        matched_images, matched_labels = Utils.match_images_to_labels(image_paths, label_paths)
        out_folder = os.path.join(dataset_folder, subfolder)
        os.makedirs(out_folder, exist_ok=True)
        Utils.write_list_txt(matched_images, os.path.join(out_folder, "images.txt"))
        Utils.write_list_txt(matched_labels, os.path.join(out_folder, "labels.txt"))

    @staticmethod
    def convert_tile_img_pth_to_basename(img_path):
            # Extract the filename from the tiled image path (e.g., 'PI_1718720450_372_Iver3069_0_0.png')
            tiled_filename = os.path.basename(img_path)
            # Extract the *original* basename from the tiled filename.
            # This removes the tile coordinates ('_0_0') and the extension ('.png').
            # Rsplit is correct here: ('PI_1718720450_372_Iver3069_0_0.png').rsplit('_', 2)[0] 
            # yields 'PI_1718720450_372_Iver3069'
            tiled_basename = tiled_filename.rsplit('_', 2)[0]
            return tiled_basename
    
    @staticmethod
    def get_all_img_lbl_pths(BASE_DIR="D:\\datasets\\tiled", SPLITS=["train", "test", "validation"]):
        # 1. Configuration
        IMAGE_EXTENSIONS = ["png", "jpg"]  # Store as simple strings for pathlib
        
        # 2. Initialize lists and ensure BASE_DIR is a Path object
        base_path = Path(BASE_DIR)
        all_image_paths = []
        all_label_paths = []
        
        # Determine the effective splits to search
        splits_to_search = SPLITS if SPLITS else [""] # If SPLITS is None/empty, search BASE_DIR directly

        # 3. Aggregate Image Paths
        # The structure is assumed to be: BASE_DIR/split_name/image_dir/*.ext
        for split in splits_to_search:
            # Construct the path pattern: BASE_DIR / split / 'images' / *.(png|jpg)
            # We assume a fixed subdirectory 'images' for images inside each split folder
            image_search_path = base_path / split / "images"
            
            # Using nested generators for a concise and memory-efficient way to collect paths
            # The '*' in the pattern is the key to globbing
            for ext in IMAGE_EXTENSIONS:
                # '**/*' is for recursive search, but here we search only one level deep
                # using `*.ext` on the target path `image_search_path`
                all_image_paths.extend(image_search_path.glob(f"*.{ext}"))


        # 4. Aggregate Label Paths
        # The structure is assumed to be: BASE_DIR/split_name/labels/*.txt
        for split in splits_to_search:
            # Construct the path pattern: BASE_DIR / split / 'labels' / *.txt
            # We assume a fixed subdirectory 'labels' for labels inside each split folder
            label_search_path = base_path / split / "labels"
            
            # Use glob to find all .txt files in the label directory
            all_label_paths.extend(label_search_path.glob("*.txt"))

        
        # 5. Convert pathlib objects to strings (if required by calling code, otherwise keep as Path objects)
        all_image_paths_str = [str(p) for p in all_image_paths]
        all_label_paths_str = [str(p) for p in all_label_paths]
        
        # 6. Assert and print checks
        print(f"Total images found in {BASE_DIR}: {len(all_image_paths_str)}")
        print(f"Total labels found: {len(all_label_paths_str)}")
        assert len(all_image_paths_str) == len(all_label_paths_str), "Mismatch between image and label counts."
        
        return all_image_paths_str, all_label_paths_str
    
    @staticmethod
    def check_txt_file_vs_images(tile_img_list_path, tile_img_dir):
        """
        Performs an integrity check by comparing image basenames (without extensions) 
        found in the 'tiled/images' folder against those listed in 'tiled/images.txt'.
        
        Returns True if the sets of basenames match exactly, False otherwise.
        """
        name = os.path.dirname(tile_img_dir)
        print(f"--- Checking Batch: {name} (Ignoring Extensions) ---")

        # Step 1: Check for existence of the master list file
        if not os.path.exists(tile_img_list_path):
            print(f"Integrity Check FAILED: Master list file not found at: {tile_img_list_path}")
            return False
            
        # Helper function to remove the extension (e.g., 'file.jpg' -> 'file')
        def strip_extension(filename):
            return os.path.splitext(filename)[0]

        # Step 2: Get basenames (NO EXTENSION) of images present in the folder
        image_paths_on_disk = glob.glob(os.path.join(tile_img_dir, "*.png")) + \
                            glob.glob(os.path.join(tile_img_dir, "*.jpg"))
                            
        # Convert full paths to a set of basenames without extensions
        tiled_imgs_basenames_no_ext = {
            strip_extension(os.path.basename(f)) for f in image_paths_on_disk
        }
        
        # Step 3: Get basenames (NO EXTENSION) of images listed in the text file
        existing_tiled_txt = Utils.read_list_txt(tile_img_list_path)
        existing_tiled_txt_basenames_no_ext = {
            strip_extension(os.path.basename(x)) for x in existing_tiled_txt
        }

        # Step 4: Perform the set difference comparison
        in_txt_not_in_folder = existing_tiled_txt_basenames_no_ext.difference(tiled_imgs_basenames_no_ext)
        in_folder_not_in_txt = tiled_imgs_basenames_no_ext.difference(existing_tiled_txt_basenames_no_ext)
        
        # Step 5: Report results
        if not in_txt_not_in_folder and not in_folder_not_in_txt:
            print(f"Integrity Check PASSED: {len(tiled_imgs_basenames_no_ext)} core files match the list.")
            return True
        else:
            print(f"Integrity Check FAILED for {name}!")
            print(f"   {len(in_txt_not_in_folder)} basenames in text file not found in folder.")
            print(f"   {len(in_folder_not_in_txt)} basenames in folder not listed in text file.")
            
            # Print a few examples for debugging
            if in_txt_not_in_folder:
                print(f"  Example text file exclusives: {list(in_txt_not_in_folder)[:3]}")
            if in_folder_not_in_txt:
                print(f"  Example folder exclusives: {list(in_folder_not_in_txt)[:3]}")
            return False

    @staticmethod
    def zip_folder(folder_path, archive_name=None):
        """
        Zips a specified folder path, placing the resulting ZIP file 
        in the parent directory of the folder.

        This function is generalized and can be used on any directory structure.

        Args:
            folder_path (str): The full path to the directory to be zipped.
            archive_name (str, optional): The desired base name for the output archive
                                        (e.g., 'my_data_archive'). If None, the 
                                        ZIP file is named after the folder being zipped.

        Returns:
            str or None: The full path to the created ZIP file, or None if zipping fails.
        """
        
        # 1. Input Validation
        if not os.path.isdir(folder_path):
            print(f"Error: Folder not found or is not a directory: {folder_path}")
            return None

        # 2. Path Setup for shutil.make_archive
        
        # Get the parent directory (this will be the root_dir)
        root_dir = os.path.dirname(folder_path)
        
        # Get the name of the folder being zipped (this will be the base_dir)
        base_dir = os.path.basename(folder_path)
        
        # Determine the name of the resulting archive file
        if archive_name is None:
            archive_name = base_dir

        # The full path and base name for the archive file (without the .zip extension)
        # The archive will be placed in root_dir
        archive_base_name = os.path.join(root_dir, archive_name)
        
        print(f"Zipping content of '{base_dir}' into '{archive_base_name}.zip'")
        
        # 3. Zipping Logic
        try:
            # shutil.make_archive zips the directory 'base_dir' found inside 'root_dir'
            # and names the resulting archive using 'archive_base_name'.
            zip_filepath = shutil.make_archive(
                base_name=archive_base_name, 
                format='zip', 
                root_dir=root_dir, 
                base_dir=base_dir
            )
            print(f"Successfully created: {zip_filepath}")
            return zip_filepath
        except Exception as e:
            print(f"Error zipping folder '{folder_path}': {e}")
            return None

    @staticmethod
    def list_tiled_set(full_image_paths, all_tiled_image_paths, all_tiled_label_paths):
        """
        Creates a test set of images that correspond to the image names input.
        """
        # Step 1: Extract and clean basenames from the full image paths
        # Create a set of basenames (e.g., 'PI_1718720450_372_Iver3069') for fast lookup.
        full_image_basenames = set()
        for full_path in full_image_paths:
            # Get the filename (e.g., 'PI_1718720450_372_Iver3069.png')
            basename_with_ext = os.path.basename(full_path)
            # Remove the extension (e.g., '.png')
            basename = os.path.splitext(basename_with_ext)[0]
            full_image_basenames.add(basename) 
            # For your data, this set will contain:
            # {'PI_1718720450_372_Iver3069', 'PI_1718720490_371_Iver3069', ...}

        tiled_set_images = []
        tiled_set_labels = []

        # Step 2: Iterate through all tiled paths and match them to the original basenames
        for img_path, lbl_path in zip(all_tiled_image_paths, all_tiled_label_paths):

            tiled_basename = Utils.convert_tile_img_pth_to_basename(img_path)
            # Check if this cleaned basename exists in our set of original basenames.
            if tiled_basename in full_image_basenames: # <-- THE CRITICAL CHANGE
                tiled_set_images.append(img_path)
                tiled_set_labels.append(lbl_path)
        
        # After the fix, for your provided data, len(tiled_images) will be 6.
        return tiled_set_images, tiled_set_labels
    
    def delete_folders_list(folder_list):
        print(f"Found {len(folder_list)} folders to delete.")

        if folder_list:
            # Safely iterate and remove each folder
            for folder in folder_list:
                try:
                    # Check if the path exists before attempting to delete
                    if os.path.isdir(folder):
                        shutil.rmtree(folder)
                        print(f"Deleted: {folder}")
                    else:
                        print(f"Skipped: {folder} (Not a directory)")
                except Exception as e:
                    print(f"Error deleting {folder}: {e}")
        else:
            print("No existing folders found.")
    
    @staticmethod
    def list_full_set(set_image_names, all_image_paths, all_label_paths):
        """
        Creates a test set of images that correspond to the image names input.

        Args:
            original_test_image_paths (list): List of image paths from the original test set.
            all_tiled_image_paths (list): List of all image paths from the entire tiled set.
            all_tiled_label_paths (list): List of all label paths from the entire tiled set.

        Returns:
            tuple: A tuple containing lists of matched tiled image paths and label paths.
        """
        
        set_images = []
        set_labels = []

        # Step 2: Iterate through all tiled paths and match them to the original basenames
        # This is the core matching step. We iterate through the tiled set once.
        for img_path, lbl_path in zip(all_image_paths, all_label_paths):
            # Extract the basename from the tiled image path.
            basename = os.path.basename(img_path).split(".")[0]
            
            # Check if this basename exists in our set of original basenames.
            if basename in set_image_names:
                set_images.append(img_path)
                set_labels.append(lbl_path)
        
        return set_images, set_labels
    

    @staticmethod
    def convert_png_to_highest_quality_jpeg(png_path, jpg_path):
        """
        Converts a single PNG image to a JPEG with the highest quality settings.
        Prints only on failure.
        """
        try:
            # 1. Open the image
            img = PIL.Image.open(png_path)
            
            # 2. Convert to RGB to discard the alpha channel (necessary for JPEG)
            if img.mode in ('RGBA', 'P'):
                # Create a white background for the image to blend onto
                background = PIL.Image.new('RGB', img.size, (255, 255, 255))
                # Paste the PNG onto the background (the alpha channel is used here)
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
                
            # 3. Save as JPEG with the highest quality and disabled subsampling
            img.save(
                jpg_path, 
                'JPEG', 
                quality=95, 
                subsampling=0
            )
            
            # Success: RETURN silently, do NOT print.
            return True
            
        except Exception as e:
            # Failure: PRINT the image name and error
            tqdm.write(f"❌ Failed to convert {os.path.basename(png_path)}: {e}")
            return False