from ultralytics import SAM
from PIL import Image
# Load the SAM model once at the module level
# you can change "sam_b.pt" to the path of your SAM model file
# download here: https://github.com/ultralytics/assets/releases/download/v8.3.0/sam_b.pt
sam_model = SAM("sam_b.pt")

class SamOverlays:
    @staticmethod
    def sam_model(img_path, bboxes=None, points=None):
        im = Image.open(img_path)
        imw, imh = im.size
        if bboxes:
            boxes = [[x, y, x + w, y + h] for x, y, w, h in bboxes]
            results = sam_model.predict(
                source=img_path,
                boxes=boxes,
                multimask_output=False,
                box_threshold=0.1,
                point_threshold=0.5,
                points_per_side=32,
                output_type="pil",
            )
        elif points:
            if isinstance(points[0], (list, tuple)):
                pts = [(int(x), int(y)) for x, y in points]
            else:
                pts = [(int(points[0]), int(points[1]))]
            point_labels = [1] * len(pts)
            results = sam_model.predict(
                source=img_path,
                points=pts,
                point_labels=point_labels,
                multimask_output=False,
                box_threshold=0.1,
                point_threshold=0.5,
                points_per_side=32,
                output_type="pil",
            )
        else:
            raise ValueError("Either bboxes or points must be provided.")
        return results
