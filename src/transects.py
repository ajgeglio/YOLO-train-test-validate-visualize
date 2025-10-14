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
    def calculate_filtered_biomass(group, OUTLIER_THRESHOLD = 80):
        """
        Calculates total biomass density (g/m^2) for an image (Filename), 
        excluding individual weights above the threshold.
        """
        # filters out individual object weights greater than the threshold before summing the remaining weights within each
        # 1. Filter the weights (only keeping values <= 80g)
        try:
            filtered_weight = group[group['box_DL_weight_g_corr'] <= OUTLIER_THRESHOLD]['box_DL_weight_g_corr']
        except KeyError:
            filtered_weight = group[group['Poly_Corr_weight_g'] <= OUTLIER_THRESHOLD]['Poly_Corr_weight_g']

        # 2. Sum the filtered weights
        total_weight_g = filtered_weight.sum()
        
        # 3. Get the constant area (using .iloc[0] is one clean way)
        image_area_m2 = group['ImageArea_m2'].iloc[0]
        
        # 4. Calculate final biomass density
        biomass_g_p_m2 = total_weight_g / image_area_m2
        
        return biomass_g_p_m2

    @staticmethod
    def biomass_transects(transect_df):
        """
        Calculate biomass density for a transect in a DataFrame filtered to the transect.               
        """
        # Determine the case-sensitive column name for 'collect_id'
        if 'collect_id' in transect_df.columns:
            id_col = 'collect_id'
        elif 'CollectID' in transect_df.columns:
            id_col = 'CollectID'
        elif 'COLLECT_ID' in transect_df.columns:
            id_col = 'COLLECT_ID'
        else:
            raise KeyError("DataFrame is missing the 'collect_id' or 'CollectID' column.")
        # Apply the custom function to the grouped data and name the new column
        biomass_df = transect_df.groupby('Filename').apply(
            Transects.calculate_filtered_biomass
        ).reset_index(name='biomass_g_p_m2')
        # 1. Create a lookup table for coordinates
        # Since Lat/Lon are constant per Filename, we can use the .first() value
        lat_lon_lookup = transect_df.groupby('Filename')[['Latitude', 'Longitude']].first().reset_index()
        # 2. Merge the biomass results with the coordinates
        biomass_transect = pd.merge(biomass_df, lat_lon_lookup, on='Filename', how='left')
        # 3. Add the collect_id column
        collect_id_lookup = transect_df[['Filename', id_col]].drop_duplicates()
        biomass_transect = pd.merge(biomass_transect, collect_id_lookup, on='Filename', how='left')
        # 4. Add the cumulative distance column
        biomass_transect = Transects.calculate_distance_along_track(biomass_transect)
        return biomass_transect

    def plot_biomass_comparison_moving_average(infer_transect, lbl_transect=None, poly_transect=None, moving_average_window=5, save_path=None):
        """
        Generates a plot of biomass density predicted and observed moving average along transect
        vs. cumulative distance for two surveys.
        """
        fig = plt.figure(figsize=(6, 4))

        # Define the transects, labels, and colors to be plotted
        # (transect_object, plot_label, line_color)
        transects_to_plot = [
            (infer_transect, 'Inferred box weight', 'blue')
        ]

        if lbl_transect is not None:
            transects_to_plot.append((lbl_transect, 'Label box weight', 'orange'))
        
        if poly_transect is not None:
            transects_to_plot.append((poly_transect, 'Polynomial weight', 'green'))

        # Loop through the transects to calculate the moving average and plot
        for transect_data, label, color in transects_to_plot:
            # Get distance (X-axis) and biomass (Y-axis)
            distance = transect_data.cumulative_distance_meters
            biomass = transect_data.biomass_g_p_m2
            
            # Calculate the moving average for biomass
            biomass_ma = biomass.rolling(
                window=moving_average_window, 
                min_periods=1
            ).mean()
            
            # Plot the moving average
            plt.plot(distance, biomass_ma, label=label, color=color)

        # Set plot aesthetics
        plt.xlabel('Distance along track (m)', fontsize=12)
        plt.ylabel('Biomass (g/m$^2$)', fontsize=12)
        plt.title(f'Biomass Moving Average (Window={moving_average_window}) Comparison Along Transect', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


    def create_observed_vs_predicted_plot(lbl_df, infer_df, save_path=None):
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
        try:
            print(f'analyzing {infer_df.collect_id.unique().item()}')
        except:
            print(f'analyzing {infer_df.CollectID.unique().item()}')
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
        fig = plt.figure(figsize=(5, 5))

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
        if save_path:
            fig.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def plot_biomass_comparison_diff(infer_transect, lbl_transect, save_path=None):
        """
        Generates a diff plot of biomass density (predicted - observed) along transect
        vs. cumulative distance for two surveys.
        """
        fig = plt.figure(figsize=(12, 6))

        # --- 1. Calculate Biomass diff (Positive Y-axis) ---
        infer_x = infer_transect.cumulative_distance_meters
        infer_y = infer_transect.biomass_g_p_m2
        lbl_x = lbl_transect.cumulative_distance_meters
        lbl_y = lbl_transect.biomass_g_p_m2
        df_infer = pd.DataFrame({'Distance': infer_x, 'Biomass': infer_y})
        df_lbl = pd.DataFrame({'Distance': lbl_x, 'Biomass': lbl_y})
        df = pd.merge_asof(df_infer.sort_values('Distance'), df_lbl.sort_values('Distance'), on='Distance', suffixes=('_infer', '_lbl'))
        df['Biomass_diff'] = df['Biomass_infer'] - df['Biomass_lbl']
        # --- 2. Plot Labeled Survey (Negative Y-axis) ---
        # Multiply the biomass values by -1 to mirror the plot

        # 1. Calculate symmetrical Y-axis limits
        max_biomass_diff = df.Biomass_diff.max() * 1.1
        min_biomass_diff = df.Biomass_diff.min() - df.Biomass_diff.min() * 0.1
        plt.ylim(min_biomass_diff, max_biomass_diff)
        
        # 2. Define the tick LOCATIONS
        # Use numpy.linspace to create evenly spaced tick positions. 
        # Adjust the '5' to change the number of ticks if needed.
        tick_locations = np.linspace(min_biomass_diff, max_biomass_diff, 7) # Example: 7 ticks
        
        # 3. Define the tick LABELS (mirroring the positive magnitude)
        # The labels are the absolute values of the locations, formatted to one decimal place
        tick_labels = [f'{abs(loc):.1f}' for loc in tick_locations]

        plt.plot(df.Distance, 
                df.Biomass_diff, 
                label='Biomass Difference', 
                color='blue',
                marker='o', 
                linestyle='-', 
                linewidth=1.5,
                markersize=4)
        plt.xlabel('Distance along track (m)', fontsize=12)
        plt.ylabel('Biomass Diff (predicted - observed) (g/m$^2$)', fontsize=12)

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
        if save_path:
            fig.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

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

