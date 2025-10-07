from math import radians, cos, sqrt
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Transects:
    @staticmethod
    def calculate_distance_along_track(transect):
        '''Simplified Euclidean Distance on Degrees
        For a very rough approximation over short distances, you can treat the degrees as Cartesian coordinates and calculate the Euclidean distance. Then, convert the distance in degrees to meters. This method is the simplest but the least accurate, as it completely ignores the Earth's curvature.

        Steps:
        Calculate the change in latitude (Δlat) and longitude (Δlon) between consecutive points.

        Apply the Pythagorean theorem on these changes.

        Convert the result to meters using conversion factors. A good approximation is that 1 degree of latitude is about 111,000 meters, and 1 degree of longitude is about 111,000 meters multiplied by the cosine of the latitude.
        '''
        # Assuming your DataFrame is named 'df'
        # Shift to get the previous point's coordinates
        transect['prev_lat'] = transect['Latitude'].shift(1)
        transect['prev_lon'] = transect['Longitude'].shift(1)

        # Calculate the difference in degrees
        transect['delta_lat'] = transect['Latitude'] - transect['prev_lat']
        transect['delta_lon'] = transect['Longitude'] - transect['prev_lon']

        # Conversion factors (approximate)
        # 1 degree of latitude is ~111.13 km
        # 1 degree of longitude is ~111.32 km * cos(latitude)
        transect['lat_to_meters'] = 111130
        transect['lon_to_meters'] = 111320 * transect['Latitude'].apply(lambda x: cos(radians(x)))

        # Calculate the distance in meters for each leg
        transect['delta_x_meters'] = transect['delta_lon'] * transect['lon_to_meters']
        transect['delta_y_meters'] = transect['delta_lat'] * transect['lat_to_meters']

        # Euclidean distance for each segment
        transect['segment_distance_meters'] = transect.apply(
            lambda row: sqrt(row['delta_x_meters']**2 + row['delta_y_meters']**2) if pd.notnull(row['delta_x_meters']) else 0,
            axis=1
        )

        # Calculate the cumulative distance
        transect['cumulative_distance_meters'] = transect['segment_distance_meters'].cumsum()

        # Clean up temporary columns
        transect = transect.drop(columns=['prev_lat', 'prev_lon', 'delta_lat', 'delta_lon', 'lat_to_meters', 'lon_to_meters', 'delta_x_meters', 'delta_y_meters'])
        return transect

    @staticmethod
    def calculate_filtered_biomass(group, OUTLIER_THRESHOLD = 30):
        """
        Calculates total biomass density (g/m^2) for an image (Filename), 
        excluding individual weights above the threshold.
        """
        # 1. Filter the weights (only keeping values <= 30g)
        try:
            filtered_weight = group[group['box_DL_weight_g_corr'] <= OUTLIER_THRESHOLD]['box_DL_weight_g_corr']
        except KeyError:
            filtered_weight = group[group['Poly_Weight_g'] <= OUTLIER_THRESHOLD]['Poly_Weight_g']

        # 2. Sum the filtered weights
        total_weight_g = filtered_weight.sum()
        
        # 3. Get the constant area (using .iloc[0] is one clean way)
        image_area_m2 = group['ImageArea_m2'].iloc[0]
        
        # 4. Calculate final biomass density
        biomass_g_p_m2 = total_weight_g / image_area_m2
        
        return biomass_g_p_m2

    @staticmethod
    def biomass_transects(assessment_df, OUTLIER_THRESHOLD=30, collect_id=None):
        """
        Calculate biomass density for each transect in the DataFrame.               
        """
        try:
            transects_df = assessment_df[assessment_df.collect_id == collect_id]
        except AttributeError:
            transects_df = assessment_df[assessment_df.CollectID == collect_id]
        # Apply the custom function to the grouped data and name the new column
        biomass_df = transects_df.groupby('Filename').apply(
            Transects.calculate_filtered_biomass
        ).reset_index(name='biomass_g_p_m2')
        # 1. Create a lookup table for coordinates
        # Since Lat/Lon are constant per Filename, we can use the .first() value
        lat_lon_lookup = transects_df.groupby('Filename')[['Latitude', 'Longitude']].first().reset_index()
        # 2. Merge the biomass results with the coordinates
        biomass_transect = pd.merge(biomass_df, lat_lon_lookup, on='Filename', how='left')
        # 3. Add the collect_id column
        try:
            collect_id_lookup = transects_df[['Filename', 'collect_id']].drop_duplicates()
            biomass_transect = pd.merge(biomass_transect, collect_id_lookup, on='Filename', how='left')
        except KeyError:
            collect_id_lookup = transects_df[['Filename', 'CollectID']].drop_duplicates()
            biomass_transect = pd.merge(biomass_transect, collect_id_lookup, on='Filename', how='left')
        # 4. Add the cumulative distance column
        biomass_transect = Transects.calculate_distance_along_track(biomass_transect)
        return biomass_transect

    def plot_biomass_comparison_mirrored(infer_transect, lbl_transect):
        """
        Generates a mirrored (bi-directional) plot of biomass density 
        vs. cumulative distance for two surveys.
        """
        plt.figure(figsize=(12, 6))

        # --- 1. Plot Inferred Survey (Positive Y-axis) ---
        plt.plot(
            infer_transect.cumulative_distance_meters, 
            infer_transect.biomass_g_p_m2, 
            label='Inferred Survey (AI Detection)', 
            marker='o', 
            linestyle='-', 
            color='black', 
            linewidth=1.5,
            markersize=4
        )
        
        # --- 2. Plot Labeled Survey (Negative Y-axis) ---
        # Multiply the biomass values by -1 to mirror the plot
        plt.plot(
            lbl_transect.cumulative_distance_meters, 
            lbl_transect.biomass_g_p_m2 * -1, 
            label='Labeled Survey (Manual QC)', 
            marker='s', 
            linestyle='-', 
            color='green', 
            linewidth=1.5,
            markersize=4
        )
        # 1. Calculate symmetrical Y-axis limits
        max_biomass = max(infer_transect.biomass_g_p_m2.max(), lbl_transect.biomass_g_p_m2.max()) * 1.1
        plt.ylim(-max_biomass, max_biomass)
        
        # 2. Define the tick LOCATIONS
        # Use numpy.linspace to create evenly spaced tick positions. 
        # Adjust the '5' to change the number of ticks if needed.
        tick_locations = np.linspace(-max_biomass, max_biomass, 7) # Example: 7 ticks
        
        # 3. Define the tick LABELS (mirroring the positive magnitude)
        # The labels are the absolute values of the locations, formatted to one decimal place
        tick_labels = [f'{abs(loc):.1f}' for loc in tick_locations]


        plt.xlabel('Distance along track (m)', fontsize=12)
        plt.ylabel('Biomass Density (g/m$^2$)', fontsize=12)

        # 4. Set both the Locator and Formatter simultaneously to suppress the Warning
        plt.yticks(tick_locations, tick_labels)
        plt.xlim(0, min(infer_transect.cumulative_distance_meters.max(), lbl_transect.cumulative_distance_meters.max()))
        

        # --- 4. Final Touches ---
        try:
            title_id = infer_transect.collect_id.iloc[0] # Use collect_id as the transect identifier
        except:
            title_id = 'Transect'

        plt.title(f'Comparative Biomass Distribution Over Distance for {title_id}', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        
        # Add a horizontal line at Y=0 to clearly separate the two plots
        plt.axhline(0, color='gray', linestyle='-', linewidth=0.8) 
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


    def create_observed_vs_predicted_plot(lbl_df, infer_df):
        """
        Generates a scatter plot comparing observed (labeled) and 
        predicted (inferred) biomass density, annotating the plot with 
        the regression line, R^2 value, and the line formula.

        Args:
            lbl_df (pd.DataFrame): DataFrame containing 'Filename' and 
                                'biomass_g_p_m2' (Observed/Labeled).
            infer_df (pd.DataFrame): DataFrame containing 'Filename' and 
                                    'biomass_g_p_m2' (Predicted/Inferred).
        """
        # 1. Merge the dataframes on Filename
        print(f'analyzing {infer_df.collect_id.unique().item()}')
        merger = pd.merge(
            lbl_df[["Filename", "biomass_g_p_m2"]], 
            infer_df[["Filename", "biomass_g_p_m2"]], 
            on="Filename", 
            how="inner", # Use inner join to ensure only matched Filenames are compared
            suffixes=["_lbl", "_infer"]
        )
        
        # 2. Extract X and Y data
        X = merger.biomass_g_p_m2_lbl.values  # Observed (X-axis)
        Y = merger.biomass_g_p_m2_infer.values # Predicted (Y-axis)
        
        if len(X) < 2:
            print("Error: Not enough data points to perform regression or plot.")
            return

        # 3. Perform Linear Regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
        r_squared = r_value**2

        # 4. Prepare Plot
        plt.figure(figsize=(5, 5))

        # 5. Scatter Plot of Data Points
        plt.scatter(X, Y, 
                    label='Data Points', 
                    color='#1f77b4', # Muted blue
                    alpha=0.6,
                    edgecolors='w',
                    linewidths=0.5,
                    s=40) 

        # 6. Plot the Ideal 1:1 Line (y=x)
        max_val = max(X.max(), Y.max()) * 1.05
        min_val = min(X.min(), Y.min()) * 0.95
        line_range = np.linspace(min_val, max_val, 10)
        
        plt.plot(line_range, line_range, 
                label='Ideal 1:1 Line', 
                color='gray', 
                linestyle='--', 
                linewidth=1)

        # 7. Plot the Regression Line
        regression_line = slope * line_range + intercept
        plt.plot(line_range, regression_line, 
                label=f'Regression Line', 
                color='red', 
                linestyle='-', 
                linewidth=2)

        # 8. Annotate the Plot with Metrics
        formula = f'$y = {slope:.3f}x + {intercept:.3f}$'
        r2_text = f'$R^2 = {r_squared:.3f}$'
        
        # Place text in a convenient corner (e.g., top-left)
        text_x = min_val + (max_val - min_val) * 0.05 
        text_y_r2 = max_val * 0.95
        text_y_formula = max_val * 0.90
        
        plt.text(text_x, text_y_formula, formula, 
                fontsize=12, color='red', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round, pad=0.5'))
                
        plt.text(text_x, text_y_r2, r2_text, 
                fontsize=12, color='red', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round, pad=0.5'))

        # 9. Set Labels and Title
        plt.xlabel('Observed Biomass ($g/m^2$)', fontsize=12)
        plt.ylabel('Predicted Biomass ($g/m^2$)', fontsize=12)
        plt.title(f'Predicted vs. Observed Biomass Density', fontsize=14)
        
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.gca().set_aspect('equal', adjustable='box') # Force 1:1 aspect ratio
        
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()