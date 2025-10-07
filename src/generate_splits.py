import shutil
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
class GenerateSplits:
    def __init__(self, img_pth_lst=None, jsn_pth_lst=None, lbl_msk_pths=None, bbox_pths = None, mer_pths = None):
        self.img_pth_lst = img_pth_lst
        self.jsn_pth_lst = jsn_pth_lst
        self.lbl_msk_pths = lbl_msk_pths
        self.bbox_pths = bbox_pths
        self.mer_pths = mer_pths
    def create_filepath_df(self):
        img_pth_lst = self.img_pth_lst
        self.filepath_df = pd.DataFrame(img_pth_lst, columns=['image_path'])
        fn = lambda x: os.path.basename(x)
        bn = lambda x: os.path.basename(x).split(".")[0]
        self.filepath_df['Filename'] = self.filepath_df.image_path.apply(bn)
    def create_label_list_df(self):
        box_list_df = pd.DataFrame(self.bbox_pths, columns=['bbox_path'])
        mer_list_df = pd.DataFrame(self.mer_pths, columns=['mer_path'])
        box_init, mer_init = box_list_df.shape[0], mer_list_df.shape[0]
        if box_init != mer_init: print("Warning: different number of bbox and MER labels")
        fn = lambda x: os.path.basename(x)
        bn = lambda x: os.path.basename(x).split(".")[0]
        # assert set(self.label_list_df.bbox_path.apply(fn)) == set(self.label_list_df.mer_path.apply(fn))
        # box_id = box_list_df.bbox_path.apply(fn)
        box_list_df['Filename'] = box_list_df.bbox_path.apply(fn).apply(bn)
        # mer_id = mer_list_df.mer_path.apply(fn)
        mer_list_df['Filename'] = mer_list_df.mer_path.apply(fn).apply(bn)
        self.label_list_df = box_list_df.merge(mer_list_df, on = "Filename", how = "left")
        assert self.label_list_df.shape[0] == box_init
    def return_merged(self, **kwargs):
        self.create_filepath_df()
        self.create_label_list_df(**kwargs)
        self.filepath_label_df = pd.merge(self.filepath_df, self.label_list_df, on='Filename', how='outer')
        dn = lambda x: os.path.dirname(x)
        folder = list(map(dn, self.img_pth_lst))
        self.filepath_label_df['year'] = pd.Series(folder).str.extract(r'.*(20[1,2][0,1,2,3,9]).*')
        return self.filepath_label_df
    def do_train_test_valid_split(self, train_split=0.1, valid_split=0.15):
        self.return_merged()
        X,y = self.filepath_label_df.path.values, self.filepath_label_df.label_path.values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=train_split, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_split, random_state=42)
        print(f"training, testing, validation, {X_train.shape[0]}, {X_test.shape[0]},{X_valid.shape[0]}")
        return X_train, y_train, X_valid, y_valid, X_test, y_test
    def do_train_valid_split(self, valid_split=0.2):
        self.return_merged()
        X,y = self.filepath_label_df.path.values, self.filepath_label_df.label_path.values
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_split, random_state=42)
        print(f"training, validation, {X_train.shape[0]},{X_valid.shape[0]}")
        return X_train, y_train, X_valid, y_valid
    def sample_df(self, df, year, n_samples, seed=42):
        dfsamp = df[df.year == year]
        dfsamp = dfsamp.sample(n_samples, random_state=seed)
        dfsamp = dfsamp.sort_values(by="Filename")
        return dfsamp.reset_index()
    def sample_df_n_fish(self, df, year, n_samples, seed=42):
        dfsamp = df[df.year == year]
        dfsamp = dfsamp.sort_values(by="n_fish", ascending=False)
        dfsamp = dfsamp.reset_index(drop=True)
        dfsamp = dfsamp[:n_samples]
        dfsamp = dfsamp.sort_values(by="Filename")
        return dfsamp.reset_index()
    def chunk_split(self, df_list, tr_frac=0.8, va_frac=0.099, year_splits = 8, seed=42):
        ''' This will take an equal proportion of each year and then sort by image name and then
            take the first n*train_proportion for training, the second proportion for validation, last for test'''
        train_df = pd.DataFrame()
        valid_df = pd.DataFrame()
        test_df = pd.DataFrame()
        # Determine the chunk size
        for df in df_list:
            n = len(df)
            # generate index of the dataframe
            arr = np.arange(0, n, 1) 
            # Split the index array into n chunks
            chunks = np.array_split(arr, year_splits)
            for c in chunks:
                lc = len(c)
                tr_idx = c[:int(lc*tr_frac)]
                vr_idx = c[int(lc*tr_frac):int(lc*(tr_frac+va_frac))]
                te_idx = c[int(lc*(tr_frac+va_frac)):]
                train_df = pd.concat([train_df, df.iloc[tr_idx]])
                valid_df = pd.concat([valid_df, df.iloc[vr_idx]])
                test_df = pd.concat([test_df, df.iloc[te_idx]])
        print("Train", train_df.shape, "valid", valid_df.shape, "test", test_df.shape)
        return train_df.sample(frac=1, random_state=seed), test_df.sample(frac=1, random_state=seed), valid_df.sample(frac=1, random_state=seed)    
    def cpy_yr_tst(df, year, root):
        ## Making/copying to directory test sets by year
        src = os.path.join(root, "test")
        dst = os.path.join(root, "test"+str(year))
        imgs = df.Filename
        lbls = df.bbox_path.apply(lambda x: os.path.basename(x))
        n = len(imgs)
        i = 1
        for img, lbl in zip(imgs, lbls):
            im_src = os.path.join(src, "images", img)
            im_dst = os.path.join(dst, "images")
            if not os.path.exists(im_dst):
                os.makedirs(im_dst)
            shutil.copy(im_src, im_dst)
            print(f"copying {i}/{n} {year} images", end="  \r")
            lb_src = os.path.join(src,"labels", lbl)
            lb_dst = os.path.join(dst, "labels")
            if not os.path.exists(lb_dst):
                os.makedirs(lb_dst)
            shutil.copy(lb_src, lb_dst)
            print(f"copying {i}/{n} {year} labels", end="  \r")
            i+=1