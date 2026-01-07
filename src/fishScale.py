import numpy as np

class FishScale:
    """
    Provides vectorized functions for calculating geometric parameters 
    and size-to-weight relationships for objects captured by AUV cameras.
    """
    
    # --- Angular Field of View (AFOV) ---
    @staticmethod
    def angular_field_of_view(sensor_dimension_mm, focal_length_mm):
        """
        Calculates the Angular Field of View (AFOV) in degrees for a given sensor 
        dimension (height or width).
        
        Args:
            sensor_dimension_mm (float/ndarray): Sensor dimension (h or w) in mm.
            focal_length_mm (float/ndarray): Lens focal length (f) in mm.

        Returns:
            float/ndarray: AFOV in degrees.
        """
        # AFOV = 2 * arctan(dimension / (2 * f))
        return 2 * np.degrees(np.arctan(sensor_dimension_mm / (2 * focal_length_mm)))
    AFOV_func = np.vectorize(angular_field_of_view.__func__)
    
    # --- Horizontal Field of View (HFOV) ---
    @staticmethod
    def horizontal_field_of_view(working_distance_mm, sensor_width_mm, focal_length_mm):
        """
        Calculates the real-world Linear Horizontal Field of View (HFOV) in the 
        object plane at the working distance.

        Args:
            working_distance_mm (float/ndarray): Working distance (W) from lens to object in mm.
            sensor_width_mm (float/ndarray): Horizontal sensor width (H) in mm.
            focal_length_mm (float/ndarray): Lens focal length (f) in mm.

        Returns:
            float/ndarray: HFOV (the captured object width) in mm.
        """
        # HFOV_obj = (Sensor Width / Focal Length) * Working Distance
        # 
        return (sensor_width_mm / focal_length_mm) * working_distance_mm
    HFOV_func = np.vectorize(horizontal_field_of_view.__func__)

    # --- Pixel Size (PS) / Ground Sample Distance (GSD) ---
    @staticmethod
    def size_to_pixel(HFOV_mm, num_pixels):
        """
        Calculates the real-world size represented by a single pixel 
        (Pixel Size or GSD) at the working distance.

        Args:
            HFOV_mm (float/ndarray): Linear Horizontal Field of View (HFOV) in mm.
            num_pixels (float/ndarray): Number of horizontal pixels (N).

        Returns:
            float/ndarray: Pixel size (PS) in mm/pixel.
        """
        return HFOV_mm / num_pixels
    PS_func = np.vectorize(size_to_pixel.__func__)

    # --- Diagonal Length in Pixels (DL_px) ---
    @staticmethod
    def calc_DL_px(width_pixel, height_pixel):
        """
        Calculates the diagonal length of an object bounding box in pixels.

        Args:
            width_pixel (float/ndarray): Object bounding box width in pixels.
            height_pixel (float/ndarray): Object bounding box height in pixels.

        Returns:
            float/ndarray: Diagonal length (DL) in pixels.
        """
        # Pythagorean theorem: sqrt(w^2 + h^2)
        return np.sqrt((width_pixel)**2 + (height_pixel)**2)
    DL_px_func = np.vectorize(calc_DL_px.__func__)

    # --- Corrected Diagonal Length in Pixels ---
    @staticmethod
    def correct_DL_px(box_DL_px, conf_pass):
        """
        Applies a custom linear correction and a confidence factor to the 
        diagonal length in pixels. This is the first step in calibration.

        Args:
            box_DL_px (float/ndarray): Raw diagonal length in pixels.
            conf_pass (float/ndarray): A confidence/pass factor (e.g., 0 or 1).

        Returns:
            float/ndarray: Corrected diagonal length in pixels.
        """
        # Polynomial correction formula from calibration
        return ((box_DL_px - 11.533) / 0.9513) * conf_pass
    correct_DL_px_func = np.vectorize(correct_DL_px.__func__)

    # --- Diagonal Length in Millimeters (DL_mm) ---
    @staticmethod
    def calc_DL_mm(box_DL_Cor_px, PS_mm):
        """
        Converts the corrected diagonal length from pixels to real-world length in mm 
        using the calculated Pixel Size (PS).

        Args:
            box_DL_Cor_px (float/ndarray): Corrected diagonal length in pixels.
            PS_mm (float/ndarray): Pixel size (real-world size per pixel) in mm.

        Returns:
            float/ndarray: Diagonal length (DL) in mm.
        """
        # Length in mm = Length in Pixels * Pixel Size (mm/pixel)
        return box_DL_Cor_px * PS_mm
    calc_DL_mm_func = np.vectorize(calc_DL_mm.__func__)

    # --- Object Weight Calculation ---
    @staticmethod
    def calc_weight(box_DL_mm):
        """
        Calculates the estimated object weight (e.g., fish weight) from its length 
        using a standard Length-Weight relationship (W = a * L^b).

        Args:
            box_DL_mm (float/ndarray): Object diagonal length (L) in mm.

        Returns:
            float/ndarray: Estimated weight (W) in grams (g).
        """
        # Model: W = exp(-12.251) * L^3.2266. Clips length to ensure non-negative power calculation.
        safe_box_mm = np.clip(box_DL_mm, a_min=0, a_max=None)
        return safe_box_mm ** 3.2266 * np.exp(-12.251)
    calc_weight_func = np.vectorize(calc_weight.__func__)

    # --- Apply Scaling Factor (Distortion Correction) ---
    @staticmethod
    def apply_scaling(box_DL_mm, SF):
        """
        Applies a final Scaling Factor (SF) to the measured length, often used 
        to correct for lens distortion based on calibration images.

        Args:
            box_DL_mm (float/ndarray): Calculated length in mm.
            SF (float/ndarray): Scaling factor.

        Returns:
            float/ndarray: Scaled length in mm.
        """
        # Divides length by SF. Uses np.where to prevent division by zero.
        return np.where(SF != 0, box_DL_mm / SF, 0)
    apply_scaling_func = np.vectorize(apply_scaling.__func__)