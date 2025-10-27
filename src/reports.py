import os
from iou import *
from results import LBLResults
from utils import Utils
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Reports:
    @staticmethod
    def generate_summary(pred_csv_path, lbl_csv_path): 
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
    
    @staticmethod
    def save_predictions_to_csv(df, csv_path):
        """Save predictions to a CSV file, ensuring no duplicates."""
        df['detect_id'] = df.Filename + "_dt_" + df.index.astype('str')
        df = df.drop_duplicates(subset="detect_id")
        df.to_csv(csv_path, header=True)

    @staticmethod
    def save_labels_to_csv(df, csv_path):
        """Save labels to a CSV file, ensuring no duplicates."""
        df = df.drop_duplicates(subset=['cls', 'x', 'y', 'w', 'h'])
        df = df.sort_values(by="Filename")
        df['ground_truth_id'] = df['Filename'] + "_" + df.index.astype('str')
        df.to_csv(csv_path, header=True)

    @staticmethod
    def output_LBL_results(meta_path, yolo_lbl_path, substrate_path, op_path, find_closest=False):
        # This is a placeholder since the LBLResults class is not provided.
        # In a real scenario, this would import and run.
        print(f"Processing results for labels")
        output = LBLResults(meta_path, yolo_lbl_path, substrate_path, op_path)
        lblres = output.lbl_results(find_closest=find_closest)
        return lblres

    @staticmethod
    def scores_df(df_lbls, df_pred, iou_tp=0.5):
        n_ground_truth = len(df_lbls)

        # 1. Use a LEFT merge to preserve ALL predictions (df_pred)
        # Unmatched predictions will have NaN values for label-related columns.
        df_merge = df_pred.merge(df_lbls, on='Filename', suffixes=('_p', '_l'), how='left')

        # Convert columns to numeric *before* calculation
        # Use errors='ignore' so NaN values from unmatched predictions don't break the column type
        cols_to_convert = ['x_l', 'x_p', 'imw', 'y_l', 'y_p', 'imh']
        for col in cols_to_convert:
             df_merge[col] = pd.to_numeric(df_merge[col], errors='coerce')

        # 2. Calculate IoU for ALL matched pairs. Unmatched predictions will get NaN IoU.
        df_merge['iou'] = df_merge.apply(lambda row: CalculateIou().get_iou(row) if not pd.isnull(row['x_l']) else 0, axis=1)
        # For completely unmatched predictions (NaN label columns), IoU should be 0.
        # This is not strictly needed if the next step handles it, but makes the column cleaner.

        # 3. Best Match for Each Prediction (including unmatched ones)
        # Drop duplicates, keeping the best IoU detect. 
        # For unmatched predictions, the iou is 0 (or NaN), but the detect_id is unique, so they are preserved.
        df_merge = df_merge.sort_values(by='iou', ascending=False).drop_duplicates(subset=['detect_id'], keep='first')
        
        # 4. Sort by confidence to form the 'scores' base
        scores = df_merge.sort_values(by='conf', ascending=False).copy()
        
        # At this point, len(scores) should equal len(df_pred). We can now assert.
        assert len(df_pred) == len(scores), "Assertion failed: length of scores does not equal length of predictions"

        # The subsequent logic needs to correctly handle the case where a prediction's best IoU is 0 (or was NaN/null).
        
        # 5. Resolve many-to-one (GT-to-Prediction) for TP assignment
        # For unmatched predictions (IoU=0), 'ground_truth_id' is NaN, which needs to be handled
        
        # Filter for actual matches where a ground truth ID exists and IoU > 0
        actual_matches = scores.dropna(subset=['ground_truth_id']).copy()

        # Group by ground truth and detect IDs, keeping the highest IoU
        max_iou = actual_matches.groupby(["ground_truth_id", "detect_id"], as_index=False).iou.max()
        max_iou = max_iou.sort_values(by='iou', ascending=False).drop_duplicates(subset='ground_truth_id', keep='first')

        # Identify true positives
        tp_id = max_iou[max_iou.iou >= iou_tp]
        
        # 6. Assign TP/FP to the full 'scores' list
        scores['tp'] = 0
        scores.loc[scores.detect_id.isin(tp_id.detect_id), 'tp'] = 1

        # False positives are *all* predictions not marked as TP.
        scores['fp'] = np.where(scores.tp == 1, 0, 1)

        # ... (rest of your logic for PR curve calculation)
        scores['acc_tp'] = scores.tp.cumsum()
        scores['acc_fp'] = scores.fp.cumsum()
        scores['Precision'] = scores.acc_tp / (scores.acc_tp + scores.acc_fp)
        scores['Recall'] = scores.acc_tp / n_ground_truth
        
        return scores.reset_index(drop=True)

    @staticmethod
    def return_fn_df(df_lbls, df_pred, iou_tp=0.5, conf_thresh=0.1):
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
    
    @staticmethod
    def scores_df_obb(df_lbls, df_pred, iou_tp = 0.5):
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
    
    @staticmethod
    def calc_AP(scores_df, step):
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
    
    @staticmethod
    def calculate_f1(scores_df):
        # Calculate F1 scores
        return 2 * (scores_df['Precision'] * scores_df['Recall']) / (scores_df['Precision'] + scores_df['Recall'])
    
    @staticmethod
    def coco_mAP(df_lbls, df_pred):
        iou_list = np.arange(0.5, 1.0, 0.05)
        ap_list = []
        for iou_ in iou_list:
            scores_df = Reports.scores_df(df_lbls, df_pred, iou_tp = iou_)
            _, _, ap = Reports.calc_AP(scores_df, step=0.01)
            ap_list.append(ap)
        mAP = np.mean(ap_list)
        return iou_list, ap_list, mAP
    
    @staticmethod
    def plot_PR(scores_df, step='AUC'):
        # Set the font to Times New Roman
        # label1 = 
        # label2 =
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 12  # Set the font size for better readability
        rec, prec, AP = Reports.calc_AP(scores_df, step)
        plt.plot(rec, prec, marker='.', label=f"P-R Curve", drawstyle="steps", linewidth=.8, markersize=1)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.title(f"Precision Recall, mAP-50: {AP:.2f}")
        # Improve the layout
        plt.grid(True)
        plt.tight_layout()
        # Show the plot
        plt.show()
        return AP

    @staticmethod
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

    @staticmethod
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
    
    @staticmethod
    def get_metrics(curve_path):
        df = pd.read_csv(curve_path, index_col=0)
        c = df.iloc[1,0].split()[1:-1]
        c = [float(num) for num in c]
        pr = df.iloc[0,1].split()[1:-1]
        pr = [float(num) for num in pr]
        f1 = df.iloc[1,1].split()[1:-1]
        f1 = [float(num) for num in f1]
        p = df.iloc[2,1].split()[1:-1]
        p = [float(num) for num in p]
        r = df.iloc[3,1].split()[1:-1]
        r = [float(num) for num in r]
        fmax = np.max(f1)
        cmax = c[np.argmax(f1)]
        df = pd.DataFrame(np.c_[c, p, r, f1], columns=["conf", "precision", "recall", "f1"])
        df['diff'] = abs(df['precision'] - df['recall'])
        eq = df.loc[df['diff'].idxmin(), ['conf', "recall"]]
        c_eq = eq.conf
        pr_eq = eq.recall
        print(f'The confidence threshold where "precision" and "recall" are the closest is: {c_eq} @ {pr_eq}')
        print(f'The confidence threshold where "F1" is at its maximum is: {cmax} @ {fmax}')
        return df, fmax, cmax, c_eq, pr_eq