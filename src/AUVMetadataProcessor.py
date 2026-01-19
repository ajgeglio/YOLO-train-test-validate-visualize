# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
from utils import Utils, ReturnTime
import datetime
import re
import pickle

@staticmethod
def load_pickle(pickle_pth): #unpickling
    with open(pickle_pth, "rb") as fp:   
        pf = pickle.load(fp)
    return pf

@staticmethod
def dump_pickle(pickle_pth, pf): #pickling
    with open(pickle_pth, "wb") as fp:
        pickle.dump(pf, fp)

class AUVDataProcessor:
    def __init__(self, out_folder, unpacked_image_directory, metadata_directory, years):
        self.out_folder = out_folder
        self.unpacked_image_directory = unpacked_image_directory
        self.years = years
        self.metadata_directory = metadata_directory
        self.Ymmdd = Utils.return_Ymmdd()
        header_directory = os.path.join(self.metadata_directory,"*","*Metadata*.csv")
        self.headers_paths = glob.glob(header_directory)

    def update_collect_lists(self, years):
        collects_txt_pth = os.path.join(self.out_folder, f"all_collects_paths.txt")
        try:
            collects_init = load_pickle(collects_txt_pth)
        except:
            collects_init = []
        collects = []
        for year in years:
            if year == "2019":
                collects = self.list_collects_paths(f"M:\\AUV_Collects\\{year}")
            if year in ("2020", "2021"):
                collects += self.list_collects_paths(f"W:\\AUV_Collects\\{year}")
            if year == "2022":
                collects += self.list_collects_paths(f"Z:\\AUV_Collects\\{year}\\*")
            if year == "2023":
                collects += self.list_collects_paths(f"X:\\AUV_Collects\\{year}\\*")
            if year == "2024":
                collects += self.list_collects_paths(f"X:\\AUV_Collects\\{year}\\*")
                collects += self.list_collects_paths(f"W:\\AUV_Collects\\{year}\\*")
        collects_final = list(set(collects + collects_init))
        new = list(set(collects_final).difference(collects_init))
        print("found", len(new), "new collects")
        if len(new) > 0:
            dump_pickle(collects_txt_pth, collects_final)
        return collects_final
            
    def list_collects_paths(self, directory):
        pat = '([0-9]{8}_[0-9]{3}_[a-z,A-Z]+[0-9]*_[a-z,A-Z]*[0-9]*[a-z,A-Z]*)'
        paths = glob.glob(os.path.join(directory, "*"))
        collects = [p for p in paths if re.search(pat, p)]
        non_matching = [p for p in paths if not re.search(pat, p)]
        if non_matching:
            print("Non-matching paths:", non_matching)
        return collects

    def get_image_list(self, year):
        base_path = os.path.join(self.unpacked_image_directory, f"{year}_UnpackedCollects")
        patterns = [
            f"{base_path}\\*Iver*\\*PrimaryImages\\*.png",
            f"{base_path}\\*REMUS*\\*PrimaryImages\\*.png"
        ]
        images = []
        for pattern in patterns:
            images.extend(glob.glob(pattern))
        img_list = [img for img in images if ".png" in img]
        img_list = [img for img in img_list if "Thumbs" not in img]
        return img_list
    
    def create_image_df(self, image_list):
        df_images = pd.DataFrame(image_list, columns=["image_path"])
        im = lambda x: os.path.basename(x).split(".")[0]
        df_images["Filename"] = df_images.image_path.map(im) # Requires "image_path" column
        # df_images = df_images.drop_duplicates(subset="image_path")
        pat1 = r'([0-9]{8}_[0-9]{3}_[a-z,A-Z]+[0-9]*_[a-z,A-Z]*[0-9]*[a-z,A-Z]*)'
        pat2 = r'([0-9]{8}_[0-9]{3}_[a-z,A-Z]+[0-9][0-9][0-9][0-9]_[a-z,A-Z]+[0-2])'
        df_images['collect_id'] = df_images["image_path"].str.extract(pat1)
        return df_images

    def create_image_dataframe(self, year):
        # Check if the year is in the headers paths
        image_list = self.get_image_list(year)
        df_images = self.create_image_df(image_list=image_list)
        assert len(df_images) == len(image_list), f"Mismatch in image list and dataframe for year {year}"
        print(f"Images for {year}: {df_images.shape}")
        return df_images

    def combine_headers(self, year):
        # Filter header paths for the given year using a more efficient list comprehension
        year_headers = [h for h in self.headers_paths if f"{year}" in h]
        header_frames = []
        for header in year_headers:
            message = f"Processing header file: {os.path.basename(os.path.dirname(header))}"
            message_len = Utils.print_progress(message)
            # Use only necessary columns if possible to reduce memory usage
            header_df = pd.read_csv(header, low_memory=False)
            # Ensure the first column is always labeled 'Time_s'
            if header_df.columns[0] != "Time_s":
                    header_df = header_df.rename(columns={header_df.columns[0]: "Time_s"})
            header_frames.append(header_df)
        print()
        if header_frames:
            combined_header = pd.concat(header_frames, ignore_index=True)
            # Drop duplicates and sort only once after concatenation
            combined_header = combined_header.drop_duplicates(subset=['Time_s', 'Filename', 'CollectID']).copy()
            combined_header = combined_header.sort_values(by='Time_s').reset_index(drop=True)
            combined_header['combine_date'] = self.Ymmdd
            combined_header.to_csv(os.path.join(self.out_folder, f"header_{year}_combined.csv"))
            # Use groupby aggregate directly for median calculation
            combined_header[["CollectID", "Latitude", "Longitude"]].groupby("CollectID", as_index=False).median().to_csv(
                os.path.join(self.out_folder, f"median_latlon_{year}.csv"), index=False
            )
            print(f"Combined header {year}: {combined_header.shape}")
            return combined_header
        else:
            print(f"No header files found for year {year}")
            return pd.DataFrame()

    def create_unpacked_images_metatada_df(self, header_df, image_df):
        # merging and keeping only image names that are unpacked
        image_df, header_df = image_df.copy(), header_df.copy()
        df_unp = header_df[header_df.Filename.isin(image_df.Filename)].copy()
        print("shape of metatata given image Filenames", df_unp.shape)
        df_unp.loc[:,"year"] = df_unp.Time_s.apply(ReturnTime().get_Y)
        df_unp.loc[:,"month"] = df_unp.Time_s.apply(ReturnTime().get_m)
        df_unp.loc[:,"day"] = df_unp.Time_s.apply(ReturnTime().get_d)
        df_unp.loc[:,"time"] = df_unp.Time_s.apply(ReturnTime().get_t)
        cam_sys = df_unp.CollectID.apply(lambda x: x.split("_")[-1])
        ims_dict_w = {"ABS1" : 4096, "ABS2": 4096, "2G": 4096, "VCC": 4096}
        ims_dict_h = {"ABS1" : 2176, "ABS2": 3000, "2G": 3000, "VCC": 3008}
        df_unp.loc[:, "imw"] = cam_sys.apply(lambda x: ims_dict_w.get(x, None)) 
        df_unp.loc[:, "imh"] = cam_sys.apply(lambda x: ims_dict_h.get(x, None)) 
        df_unp = pd.merge(image_df, df_unp, on="Filename", how = 'left')
        df_unp = df_unp.sort_values(by='Time_s').reset_index(drop=True)
        is_duplicated = df_unp[df_unp.duplicated(subset=['Filename'], keep=False)]
        if not is_duplicated.empty:
            print("Warning: Duplicates found in unpacked images metadata")
            is_duplicated.to_csv(os.path.join(self.out_folder, f"duplicates_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)
        assert df_unp.shape[0] == df_unp.drop_duplicates(subset=['Time_s','Filename','CollectID']).shape[0], "Duplicates found in unpacked images metadata"
        return df_unp
    
    def update_unpacked_images_metadata(self, years):
        try:
            latest_file = max(glob.glob(os.path.join(self.out_folder,"all_unpacked_images_metadata_*.pkl")), key=os.path.getctime)
            all_unpacked_metadata = pd.read_pickle(latest_file)
        except ValueError:
            all_unpacked_metadata = pd.DataFrame()

        try:
            latest_file = max(glob.glob(os.path.join(self.out_folder,"all_headers_*.pkl")), key=os.path.getctime)
            all_headers = pd.read_pickle(latest_file)
        except ValueError:
            all_headers = pd.DataFrame()

        for year in self.years:
            # Combine headers for the year
            year_header = self.combine_headers(year)
            all_headers = pd.concat([all_headers, year_header])

            # Create image dataframe for the year
            year_image_df = self.create_image_dataframe(year)

            # Create unpacked images metadata
            year_metadata = self.create_unpacked_images_metatada_df(year_header, year_image_df)
            year_metadata['combine_date'] = self.Ymmdd
            print(f"Unpacked images header {year}: {year_metadata.shape}")
            year_metadata.to_csv(os.path.join(self.out_folder, f"all_unpacked_images_metadata_{year}.csv"))
            all_unpacked_metadata = pd.concat([all_unpacked_metadata, year_metadata])

        # Deduplicate by dropping the older data and save all unpacked metadata
        all_unpacked_metadata = (
            all_unpacked_metadata
            .sort_values(by=["Time_s", "combine_date"])
            .drop_duplicates(subset="Filename", keep="last")
            .reset_index(drop=True)
            )
        all_unpacked_metadata.to_pickle(os.path.join(self.out_folder, f"all_unpacked_images_metadata_{self.Ymmdd}.pkl"))

        # Deduplicate and save all headers
        all_headers = (
            all_headers
            .sort_values(by=["Time_s", "combine_date"])
            .drop_duplicates(subset="Filename", keep="last")
            .reset_index(drop=True)
            )
        all_headers.to_pickle(os.path.join(self.out_folder, f"all_headers_{self.Ymmdd}.pkl"))

        print("All unpacked:", all_unpacked_metadata.shape)
        print("All headers:", all_headers.shape)

