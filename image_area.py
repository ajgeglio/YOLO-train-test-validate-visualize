import numpy as np

class ImageArea:
    @staticmethod
    def angular_field_of_view(h, f):
        # Angular field of view in degrees
        return 2 * np.degrees(np.arctan(h / (2 * f)))
    AFOV_func = np.vectorize(angular_field_of_view.__func__)

    @staticmethod
    def horizontal_field_of_view(w, h, f):
        # Horizontal field of view (HFOV) in mm
        return w * h / f
    HFOV_func = np.vectorize(horizontal_field_of_view.__func__)

    @staticmethod
    def size_to_pixel(HFOV, N):
        # Pixel size in mm
        return HFOV / N
    PS_func = np.vectorize(size_to_pixel.__func__)

    @staticmethod
    def calc_DL_px(w_pixel, h_pixel):
        # Diagonal length in pixels
        return np.sqrt((w_pixel)**2 + (h_pixel)**2)
    DL_px_func = np.vectorize(calc_DL_px.__func__)

    @staticmethod
    def correct_DL_px(box_DL_px, conf_pass):
        # Polynomial correction for diagonal length
        return ((box_DL_px - 11.533) / 0.9513) * conf_pass
    correct_DL_px_func = np.vectorize(correct_DL_px.__func__)

    @staticmethod
    def calc_DL_mm(box_DL_Cor_px, PS_mm):
        # Convert corrected diagonal length to mm
        return box_DL_Cor_px * PS_mm
    calc_DL_mm_func = np.vectorize(calc_DL_mm.__func__)

    @staticmethod
    def calc_weight(box_DL_mm):
        # Fish weight from length (g)
        return np.where(box_DL_mm > 0, (box_DL_mm ** 3.2266 * np.exp(-12.251)), 0)
    calc_weight_func = np.vectorize(calc_weight.__func__)

    @staticmethod
    def apply_scaling(box_DL_mm, SF):
        # Apply scaling factor for correction
        return np.where(SF != 0, box_DL_mm / SF, 0)
    apply_scaling_func = np.vectorize(apply_scaling.__func__)