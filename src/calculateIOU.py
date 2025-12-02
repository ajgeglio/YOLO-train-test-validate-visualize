import numpy as np
from shapely.geometry import Polygon
from shapely.validation import explain_validity
import pandas as pd
import warnings

class CalculateIou:
    def __init__(self) -> None:
        self

    def get_iou(self, row):
        """
        Original post: https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
        Only for axis-aligned bounding boxes (not rotated)
        Calculates the Intersection over Union (IoU) of bounding boxes from a dataframe row with the lbl and predict box coordinates in adjacent columns
        This functions was adapted to read the custom scores output we create with Yolo Predict and requires a pandas dataframe with set column names
        This function was designed to run inside of pandas.DataFrame.apply(lambda row: get_iou(row)), where the dataframe is our scores dataframe

        Parameters
        ----------
        dataframe[i, columns=['x_l', 'y_l, 'w_l', 'h_l', 'x_p', 'y_p', 'w_p', 'h_p', 'im_h', 'im_w']]

        bb1 : dict
            Keys: {'x_l', 'y_l', 'w_l', 'h_l'}
            The {x, y, w, h} of a label bounding box
            If there are multiple predictions , rows in these columns may be duplicated
        bb2 : dict
            Keys: {'x_p', 'y_p', 'w_p', 'h_p'}
            The {x, y, w, h} of an associtated predicted bounding box

        Returns
        -------
        float
            in [0, 1]
        """
        im_h, im_w = row['imh'], row['imw']
        bb1x1 = row['x_l']*im_w - (row['w_l']*im_w)/2
        bb1y1 = row['y_l']*im_h - (row['h_l']*im_h)/2
        bb1x2 = row['x_l']*im_w + (row['w_l']*im_w)/2
        bb1y2 = row['y_l']*im_h + (row['h_l']*im_h)/2
        '''
            The (bb1x1, bb1y1) position is at the top left corner of the label boxes
            the (bb1x2, bb1y2) position is at the bottom right corner of the label boxes
        '''

        bb2x1 = row['x_p']*im_w - (row['w_p']*im_w)/2
        bb2y1 = row['y_p']*im_h - (row['h_p']*im_h)/2
        bb2x2 = row['x_p']*im_w + (row['w_p']*im_w)/2
        bb2y2 = row['y_p']*im_h + (row['h_p']*im_h)/2
        '''
            The (bb2x1, bb2y1) position is at the top left corner of the predicted boxes
            the (bb2x2, bb2y2) position is at the bottom right corner of the predicted boxes
        '''

        assert bb1x1 < bb1x2
        assert bb1y1 < bb1y2
        assert bb2x1 < bb2x2
        assert bb2y1 < bb2y2

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1x1, bb2x1)
        y_top = max(bb1y1, bb2y1)
        x_right = min(bb1x2, bb2x2)
        y_bottom = min(bb1y2, bb2y2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1x2 - bb1x1) * (bb1y2 - bb1y1)
        bb2_area = (bb2x2 - bb2x1) * (bb2y2 - bb2y1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou
        
class CalculateIntersection:
    """
    Calculates the intersection area of a polygon relative to a reference larger 4-sided polygon.
    Works for YOLO bounding boxes.

    Required Parameters
    ----------
    dataframe[i, columns=['Filename', 'x_f', 'y_f', 'w_f', 'h_f', 'x_b', 'y_b', 'w_b', 'h_b', 'imh', 'imw']]

    bb1 : dict
        Keys: {'x_f', 'y_f', 'w_f', 'h_f'}
        The {x, y, w, h} of a fish detection or label bounding box.
    bb2 : dict
        Keys: {'x_c', 'y_c', 'w_c', 'h_c'}
        The {x, y, w, h} of a larger bounding box (e.g., cages in the GoPro images).

    Returns
    -------
    float
        Intersection area in pixels.
    """
    def __init__(self) -> None:
        pass

    def get_intersection(self, row):
        """
        Calculates the intersection area of two polygons.

        Args:
            row (dataframe iterrow): Contains the geometry of both the label and reference bounding boxes.

        Returns:
            float: The intersection area between the two polygons, or NaN if invalid.
        """
        idx = row.name
        # Image dimensions
        imw, imh = row['imw'], row['imh']
        # Label bounding box coordinates
        bbox1 = [
            ((row['x_f'] - row['w_f'] / 2) * imw, (row['y_f'] - row['h_f'] / 2) * imh),
            ((row['x_f'] + row['w_f'] / 2) * imw, (row['y_f'] - row['h_f'] / 2) * imh),
            ((row['x_f'] + row['w_f'] / 2) * imw, (row['y_f'] + row['h_f'] / 2) * imh),
            ((row['x_f'] - row['w_f'] / 2) * imw, (row['y_f'] + row['h_f'] / 2) * imh)
        ]
        # Reference bounding box coordinates
        bbox2 = [
            ((row['x_c'] - row['w_c'] / 2) * imw, (row['y_c'] - row['h_c'] / 2) * imh),
            ((row['x_c'] + row['w_c'] / 2) * imw, (row['y_c'] - row['h_c'] / 2) * imh),
            ((row['x_c'] + row['w_c'] / 2) * imw, (row['y_c'] + row['h_c'] / 2) * imh),
            ((row['x_c'] - row['w_c'] / 2) * imw, (row['y_c'] + row['h_c'] / 2) * imh)
        ]

        try:
            # Create Shapely polygons
            label_box = Polygon(bbox1)
            cage_box = Polygon(bbox2)

            # Validate geometries
            if not label_box.is_valid:
                raise ValueError(f"Invalid label geometry: {explain_validity(bbox1)}")
            if not cage_box.is_valid:
                raise ValueError(f"Invalid cage geometry: {explain_validity(bbox2)}")

            # Calculate intersection of label inside cage
            intersection = label_box.intersection(cage_box)
            return intersection.area / label_box.area if intersection.is_valid else np.nan
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            return np.nan

class CalculateIouOBB:
    """
    google prompt: "calculate IOU from 4 corners oriented bounding boxes python"
    Only for YOLO oriented bounding boxes
    Calculates the Intersection over Union (IoU) of bounding boxes from a dataframe row with the lbl and predict box coordinates in adjacent columns
    This functions was adapted to read the custom scores output we create with Yolo Predict and requires a pandas dataframe with set column names
    This function was designed to run inside of pandas.DataFrame.apply(lambda row: get_iou(row)), where the dataframe is our scores dataframe

    Parameters
    ----------
    dataframe[i, columns=['x1_l', 'y1_l', 'x2_l', 'y2_l', 'x3_l', 'y3_l', 'x4_l', 'y4_l', 'x1_p', 'y1_p', 'x2_p', 'y2_p', 'x3_p', 'y3_p', 'x4_p', 'y4_p','x', 'y', 'w', 'h', 'r', 'im_h', 'im_w']]

    bb1 : dict
        Keys: {'x_l', 'y_l', 'w_l', 'h_l'}
        The {x, y, w, h} of a label bounding box
        If there are multiple predictions , rows in these columns may be duplicated
    bb2 : dict
        Keys: {'x_p', 'y_p', 'w_p', 'h_p'}
        The {x, y, w, h} of an associtated predicted bounding box

    Returns
    -------
    float
        in [0, 1]
    """
    def __init__(self) -> None:
        self

    def get_iou(self, row):
        
        """Calculates the IoU of two oriented bounding boxes.

        Args:
            row (dataframe iterrow): has the geometry of both the prediction and the label
            bbox1 (list): A list of 8 coordinates representing the 4 corners of the first OBB.
            bbox2 (list): A list of 8 coordinates representing the 4 corners of the second OBB.

        Returns:
            float: The IoU value between the two OBBs.
        """
        imw, imh = row['imw'], row['imh']
        x1_l, y1_l, x2_l, y2_l, x3_l, y3_l, x4_l, y4_l = row['x1_l'], row['y1_l'], row['x2_l'], row['y2_l'], row['x3_l'], row['y3_l'], row['x4_l'], row['y4_l']
        x1_p, y1_p, x2_p, y2_p, x3_p, y3_p, x4_p, y4_p = row['x1_p'], row['y1_p'], row['x2_p'], row['y2_p'], row['x3_p'], row['y3_p'], row['x4_p'], row['y4_p']
        bbox1 = [(x1_l*imw, y1_l*imh), (x2_l*imw, y2_l*imh), (x3_l*imw, y3_l*imh), (x4_l*imw, y4_l*imh)]
        bbox2 = [(x1_p*imw, y1_p*imh), (x2_p*imw, y2_p*imh), (x3_p*imw, y3_p*imh), (x4_p*imw, y4_p*imh)]
        # Create Shapely polygons from the bounding box coordinates
        poly1 = Polygon(bbox1)
        poly2 = Polygon(bbox2)

        # Calculate the intersection and union areas
        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - intersection_area

        # Calculate and return the IoU
        return intersection_area / union_area  