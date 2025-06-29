import os
from iou import *
from utils import Utils
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Reports:
    def __init__(self) -> None:
        self

    def generate_summary(self, pred_csv_path, lbl_csv_path): 
        ''' 
        Input is the path where a test predict run output is stored. Meaning a label and predict csv was generated.
        These csvs are generated with the Yolo predict loop. 
        If there are only inference, then scores cannot be calculated.
        '''
        df_pred = pd.read_csv(pred_csv_path, index_col=0)
        df_lbls = pd.read_csv(lbl_csv_path, index_col=0)
        tot_pred = len(df_pred.drop_duplicates())
        mf_imgs = list(set(df_lbls.Filename).difference(set(df_pred.Filename)))
        gobyim = list(set(df_lbls[df_lbls.cls == 0].Filename))
        alewim = list(set(df_lbls[df_lbls.cls == 1].Filename))
        gobypr = list(set(df_pred[df_pred.cls == 0].Filename))
        alewpr = list(set(df_pred[df_pred.cls == 1].Filename))
        fn_whol_img = df_lbls[df_lbls.Filename.isin(mf_imgs)].shape[0]
        fp_extra = max(0, len(df_pred.Filename)-len(df_lbls.Filename))
        print(f"\n{os.path.dirname(pred_csv_path)}: summary")
        print("------------------------")
        print(f"Total number of test image labels with fish present", len(df_lbls.Filename.unique()))
        print(f"{len(gobyim)} labels with goby and {len(alewim)} labels with alewife")
        print(f"Total number of test image labels with predicted fish", len(df_pred.Filename.unique()))
        print(f"{len(gobypr)} images predicted with goby and {len(alewpr)} images predicted with alewife")
        print(f"there are", fn_whol_img, "fish labels where the prediction missed whole image")
        print(f"Total number of fish in test set", len(df_lbls), "(Ground truths)")
        print(f"Total number of fish predicted: {tot_pred:,}, at a min confidence: {df_pred.conf.min():0.2f}")
        return df_pred, df_lbls

    def scores_df(self, df_lbls, df_pred, iou_tp=0.5):
        # https://kharshit.github.io/blog/2019/09/20/evaluation-metrics-for-object-detection-and-segmentation
        n_ground_truth = len(df_lbls)
        df_merge = df_pred.merge(df_lbls, on='Filename', suffixes=('_p', '_l'), how='outer').dropna()
        # Calculate pixel distance
        df_merge[['x_l', 'x_p', 'imw', 'y_l', 'y_p', 'imh']] = df_merge[['x_l', 'x_p', 'imw', 'y_l', 'y_p', 'imh']].apply(pd.to_numeric, errors='coerce')
        df_merge['dist'] = np.sqrt(((df_merge.x_l - df_merge.x_p) * df_merge.imw) ** 2 + ((df_merge.y_l - df_merge.y_p) * df_merge.imh) ** 2)
        # Calculate IoU
        df_merge['iou'] = df_merge.apply(lambda row: CalculateIou().get_iou(row), axis=1)
        # Drop duplicates, keeping the best IoU detect
        df_merge = df_merge.sort_values(by='iou', ascending=False).drop_duplicates(subset=['detect_id'], keep='first')
        # Sort by confidence
        scores = df_merge.sort_values(by='conf', ascending=False)
        # Group by ground truth and detect IDs, keeping the highest IoU
        max_iou = scores.groupby(["ground_truth_id", "detect_id"], as_index=False).iou.max()
        max_iou = max_iou.sort_values(by='iou', ascending=False).drop_duplicates(subset='ground_truth_id', keep='first')
        # Identify true positives
        tp_id = max_iou[max_iou.iou >= iou_tp]
        tp_idx = scores[scores.detect_id.isin(tp_id.detect_id)].index
        scores['tp'] = 0
        scores.loc[tp_idx, 'tp'] = 1
        # Identify false positives
        scores['fp'] = np.where(scores.tp == 1, 0, 1)
        # Calculate cumulative sums for true positives and false positives
        scores['acc_tp'] = scores.tp.cumsum()
        scores['acc_fp'] = scores.fp.cumsum()
        # Calculate precision and recall
        scores['Precision'] = scores.acc_tp / (scores.acc_tp + scores.acc_fp)
        scores['Recall'] = scores.acc_tp / n_ground_truth
        assert len(df_pred) == len(scores)
        return scores.reset_index(drop=True)

    def return_fn_df(self, df_lbls, df_pred, iou_tp=0.5, conf_thresh=0.2):
        # Merge labels and predictions based on Filename and confidence threshold
        df_merge = df_lbls.merge(df_pred[df_pred.conf >= conf_thresh], on='Filename', suffixes=('_l', '_p'), how='left')
        # Calculate pixel distance
        df_merge['dist'] = np.sqrt(((df_merge.x_l - df_merge.x_p) * df_merge.imw) ** 2 + ((df_merge.y_l - df_merge.y_p) * df_merge.imh) ** 2)
        # Calculate IoU
        df_merge['iou'] = df_merge.apply(lambda row: CalculateIou().get_iou(row) if not pd.isna(row.x_p) else np.nan, axis=1)
        # Sort by IoU and remove duplicates
        df_merge = df_merge.sort_values(by='iou', ascending=False).drop_duplicates(subset=['ground_truth_id'], keep='first')
        # Ensure all ground truth IDs are accounted for
        missing_ids = set(df_lbls.ground_truth_id) - set(df_merge.ground_truth_id)
        if missing_ids:
            missing_df = df_lbls[df_lbls.ground_truth_id.isin(missing_ids)]
            missing_df = missing_df.assign(iou=np.nan, dist=np.nan, conf=np.nan, tp=0, fn=1, acc_tp=0, acc_fn=1)
            df_merge = pd.concat([df_merge, missing_df], ignore_index=True)
        # Sort by confidence
        fn_df = df_merge.sort_values(by='conf', ascending=False)
        # Determine true positives
        tp_idx = fn_df[fn_df.iou >= iou_tp].index
        fn_df['tp'] = 0
        fn_df.loc[tp_idx, 'tp'] = 1
        # Determine false negatives
        fn_df['fn'] = np.where(fn_df.tp == 1, 0, 1)
        # Cumulative sums for true positives and false negatives
        fn_df['acc_tp'] = fn_df.tp.cumsum()
        fn_df['acc_fn'] = fn_df.fn.cumsum()
        # Ensure all ground truth IDs are accounted for
        assert len(df_lbls) == len(fn_df)
        return fn_df.reset_index(drop=True)
    
    def scores_df_obb(self, df_lbls, df_pred, iou_tp = 0.5):
        # https://kharshit.github.io/blog/2019/09/20/evaluation-metrics-for-object-detection-and-segmentation
        n_ground_truth = len(df_lbls)
        # df_pred = df_pred[df_pred.conf>=thresh]
        df_merge = df_pred.merge(df_lbls, on='image_id', suffixes=('_p', '_l'), how='outer').dropna()
        print(df_merge.head())
        # distance calculation
        # df_merge['dist'] = np.sqrt(list(((df_merge.x_l - df_merge.x_p)*df_merge.im_w)**2 + ((df_merge.y_l - df_merge.y_p)*df_merge.im_h)**2))
        # calulating iou, function imported from calculate_iou.py
        df_merge['iou'] = df_merge.apply(lambda row: CalculateIouOBB().get_iou(row), axis=1)
        # making sure to drop the duplicate detects, only keeping the best iou detect by sorting by iou first 
        df_merge = df_merge.sort_values(by='iou', ascending=False)
        df_merge = df_merge.drop_duplicates(subset=['detect_id'], keep='first')
        # then sort by confidence as explained in the link
        scores = df_merge.sort_values(by='conf', ascending=False)
        ## If there are multiple detects for the same ground truth, only 1 is a TP
        ## Here i use pandas "group-by" to get all associated detects to given ground truth (this why is is more accurate)
        max_iou = scores.groupby(["ground_truth_id", "detect_id"], as_index=False).iou.max()
        max_iou = max_iou.sort_values(by='iou', ascending=False)
        max_iou = max_iou.drop_duplicates(subset='ground_truth_id', keep='first')
        ## only looking at boxes with iou>=0.5 or kwarg specified (the rest will be called FP)
        tp_id = max_iou[max_iou.iou>=iou_tp]
        tp_idx = scores[scores.detect_id.isin(tp_id.detect_id)].index
        scores['tp'] = 0
        scores.loc[tp_idx, 'tp'] = 1
        ## fp is simpler, now that we determined which are TP, the rest are FP
        # fp = np.where((scores.iou < iou_tp), 1, 0)
        scores['fp'] = np.where((scores.tp == 1), 0, 1)
        # cumulative sums for tp/fp and calculation of precision recall curve
        scores['acc_tp'] = scores.tp.cumsum()
        scores['acc_fp'] = scores.fp.cumsum()
        scores['Precision'] = scores.acc_tp/(scores.acc_tp + scores.acc_fp)
        scores['Recall'] = scores.acc_tp/n_ground_truth
        return scores.reset_index(drop=True)
    
    def calc_AP(self, scores_df, step):
        rec = scores_df.Recall.values
        prec = scores_df.Precision.values
        idx = Utils().evenly_spaced_indices(rec, step)
        rec_ = rec[idx]
        prec_ = prec[idx]
        prec_interp = [prec_[0]] + [max(prec_[i+1:]) for i in range(len(prec_)-1)]
        if step == 'AUC':
            AP = np.sum([(rec_[i+1] - rec_[i])*(prec_interp[i+1]) for i in range(len(rec_)-1)])
        else:
            AP = np.sum(prec_interp)/((1/step)+1)
        return rec_, prec_interp, AP
    
    def coco_mAP(self, df_lbls, df_pred):
        iou_list = np.arange(0.5, 1.0, 0.05)
        ap_list = []
        for iou_ in iou_list:
            scores_df = self.scores_df(df_lbls, df_pred, iou_tp = iou_)
            _, _, ap = self.calc_AP(scores_df, step=0.01)
            ap_list.append(ap)
        mAP = np.mean(ap_list)
        return iou_list, ap_list, mAP
    
    def plot_PR(self, scores_df, step='AUC'):
        # Set the font to Times New Roman
        # label1 = 
        # label2 =
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12  # Set the font size for better readability
        rec, prec, AP = self.calc_AP(scores_df, step)
        plt.plot(rec, prec, marker='.', label=f"P-R Curve", drawstyle="steps", linewidth=.8, markersize=1)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.title(f"Precision Recall, AP: {AP:.2f}")
        # Improve the layout
        plt.grid(True)
        plt.tight_layout()
        # Show the plot
        plt.show()
        return AP
  
    def plot_epoch_time(df1, df2=pd.DataFrame(),title=None, lbl1=None, lbl2=None):
        tdt = lambda x: datetime.datetime.fromtimestamp(x)
        fig, ax = plt.subplots(1, figsize=(10,5))
        et1 = df1['EpochTime']
        c1 = np.ones(len(et1))
        dt1 = et1.apply(tdt).values
        dt1 = pd.DataFrame([dt1,c1]).T
        dt1 = dt1.set_index(0)
        counts = dt1.groupby(pd.Grouper(freq='1D')).count()
        count_values = [list for sublist in counts.values for list in sublist]
        ax.bar(counts.index, count_values, edgecolor='k', label=lbl1)
        if not df2.empty:
            et2 = df2['EpochTime']
            c2 = np.ones(len(et2))
            dt2 = et2.apply(tdt).values
            dt2 = pd.DataFrame([dt2,c2]).T
            dt2 = dt2.set_index(0)
            counts2 = dt2.groupby(pd.Grouper(freq='1D')).count()
            count_values2 = [list for sublist in counts2.values for list in sublist]
            ax.bar(counts2.index, count_values2, edgecolor='k', alpha=0.5, label=lbl2)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
    
    def plot_collect_distribution(df1, df2=pd.DataFrame(),title=None, lbl1=None, lbl2=None, ylabel=None, w=10, h=7):
        fig, ax = plt.subplots(1, figsize=(w,h))
        counts1 = df1.groupby(by='collect_id').count()["filename"]
        ax.bar(counts1.index, counts1.values, edgecolor='k', label=lbl1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        if not df2.empty:
            counts2 = df2.groupby(by='collect_id').count()["filename"]
            ax.bar(counts2.index, counts2.values, edgecolor='k', alpha=0.5, label=lbl2)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title(title)
        ax.legend()
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        return  