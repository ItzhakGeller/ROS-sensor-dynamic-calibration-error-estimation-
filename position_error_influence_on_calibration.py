import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import os


class SensorCalibrationAnalyzer:
    def __init__(self, excel_file_path, sheet_name):
        self.excel_file = excel_file_path
        self.sheet_name = sheet_name
        self.data = self.load_excel_data()
        if self.data is not None:
            self.increment = self.detect_increment()
            self.calibrations = {}
            self.errors = {}
            self.adjusted_errors = {}
        else:
            # Handle the case where data loading fails
            print(
                f"Warning: Data loading failed for sheet '{self.sheet_name}'. Initializing with empty/default attributes."
            )
            self.increment = 0.2  # Default or None
            self.calibrations = {}  # Ensure these are initialized even if data is None
            self.errors = {}
            self.adjusted_errors = {}

    def load_excel_data(self):
        """Load Excel data from specified sheet, columns A,B starting from row 2"""
        try:
            # Read Excel file, specific sheet, skip first row (header), use columns A,B
            df = pd.read_excel(
                self.excel_file,
                sheet_name=self.sheet_name,
                skiprows=1,  # Skip first row
                usecols=[0, 1],  # Columns A,B (0,1 in 0-indexed)
                names=["distance", "sensor_reading"],
            )

            # Remove any rows with NaN values
            df = df.dropna()

            # Sort by distance
            df = df.sort_values("distance").reset_index(drop=True)

            print(f"Loaded {len(df)} data points from sheet '{self.sheet_name}'")
            print(
                f"Distance range: {df['distance'].min():.2f} - {df['distance'].max():.2f} mm"
            )
            print(
                f"Sensor reading range: {df['sensor_reading'].min():.0f} - {df['sensor_reading'].max():.0f}"
            )

            # DEBUG: Print first 10 rows to verify data
            print("\nDEBUG - First 10 data points:")
            print("Distance | Sensor Reading")
            print("-" * 25)
            for i in range(min(10, len(df))):
                print(
                    f"{df.iloc[i]['distance']:8.3f} | {df.iloc[i]['sensor_reading']:12.0f}"
                )

            # DEBUG: Check increments
            increments = np.diff(df["distance"].values)
            print(f"\nDEBUG - First 10 increments: {increments[:10]}")
            print(f"DEBUG - Median increment: {np.median(increments):.4f}")

            return df

        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return None

    def detect_increment(self):
        """Detect data increment from the dataset"""
        if self.data is None or len(self.data) < 2:
            return 0.2  # default

        increments = np.diff(self.data["distance"].values)
        detected_increment = round(np.median(increments), 3)
        print(f"Detected data increment: {detected_increment} mm")
        return detected_increment

    def _find_closest_actual_points(self, target_distances):  # Renamed
        """Find closest available actual data points to target distances"""
        calibration_points = {}
        for target in target_distances:
            closest_idx = np.argmin(np.abs(self.data["distance"] - target))
            actual_distance = self.data["distance"].iloc[closest_idx]
            sensor_value = self.data["sensor_reading"].iloc[closest_idx]
            calibration_points[target] = {
                "actual_distance": actual_distance,
                "sensor_reading": sensor_value,
            }
        return calibration_points

    def calculate_calibration_parameters(self, points):
        """Calculate A, B, C parameters using the exponential decay model.
        Assumes calibration distances X1_calib, X2_calib, X3_calib are 1.0, 2.0, 3.0 respectively.
        The 'points' argument provides the sensor readings S1, S2, S3 that correspond
        to (potentially offset) target distances, but the A,B,C math uses X_calib=1,2,3.
        """
        try:
            # Extract sensor readings S1, S2, S3 from the 'points' input
            # distances in 'points' keys are the target distances (e.g., 1.0, 2.0, 3.0 or 1.2, 2.2, 3.2 etc.)
            sorted_target_distances = sorted(points.keys())
            S1 = points[sorted_target_distances[0]]["sensor_reading"]
            S2 = points[sorted_target_distances[1]]["sensor_reading"]
            S3 = points[sorted_target_distances[2]]["sensor_reading"]

            # Use fixed X values for calibration formulas as per requirement
            X1_calib, X2_calib, X3_calib = 1.0, 2.0, 3.0

            # Primary method: B = log((S1-S2)/(S2-S3))
            # This requires S1 > S2 > S3 for B to be real and positive.
            val_S1_S2 = S1 - S2
            val_S2_S3 = S2 - S3

            params_calculated = False

            if (
                val_S1_S2 > 1e-9 and val_S2_S3 > 1e-9
            ):  # Avoid division by zero and ensure S1>S2 and S2>S3
                ratio_for_exp_neg_B = (
                    val_S2_S3 / val_S1_S2
                )  # This should be exp(-B * deltaX_calib), deltaX_calib=1
                if (
                    ratio_for_exp_neg_B > 1e-9 and ratio_for_exp_neg_B < 1.0
                ):  # Ensures B > 0
                    B = -np.log(ratio_for_exp_neg_B)

                    denominator_A = np.exp(-B * X1_calib) - np.exp(-B * X3_calib)
                    if abs(denominator_A) > 1e-9:
                        A = (S1 - S3) / denominator_A
                        C = S1 - A * np.exp(-B * X1_calib)
                        # print(f"Debug: Primary B calc: A={A:.3f}, B={B:.6f}, C={C:.3f}")
                        return {"A": A, "B": B, "C": C}
                    else:
                        print(
                            f"Warning (Primary): Denominator for A is near zero. S1={S1:.0f},S2={S2:.0f},S3={S3:.0f}, B={B:.4f}"
                        )
                else:
                    print(
                        f"Warning (Primary): Ratio for exp(-B) ({ratio_for_exp_neg_B:.3f}) is not between 0 and 1. S1={S1:.0f}, S2={S2:.0f}, S3={S3:.0f}."
                    )
            else:
                print(
                    f"Warning (Primary): Sensor readings S1,S2,S3 ({S1:.0f}, {S2:.0f}, {S3:.0f}) are not suitable for primary B calculation (not strictly monotonic decreasing or S1-S2 or S2-S3 is too small)."
                )

            # Fallback method: Estimate C first, then B, then A.
            # This method assumes (S2-C)^2 = (S1-C)(S3-C) because X_calib are equidistant.
            print(
                "Note: Attempting fallback parameter calculation by estimating C first."
            )
            denominator_C_est = S1 + S3 - 2 * S2
            if abs(denominator_C_est) < 1e-9:  # Avoid division by zero
                print(
                    "Warning (Fallback): Denominator for C_est is near zero (S1, S2, S3 may be in arithmetic progression). Exponential model may not fit well."
                )
                return None

            C_est = (S1 * S3 - S2 * S2) / denominator_C_est

            val_S1_minus_C_est = S1 - C_est
            val_S2_minus_C_est = S2 - C_est

            if (
                val_S1_minus_C_est <= 1e-9 or val_S2_minus_C_est <= 1e-9
            ):  # Need S1-C > 0 and S2-C > 0 for log
                print(
                    f"Warning (Fallback): Estimated C ({C_est:.2f}) leads to non-positive or very small (S-C) values. S1-C={val_S1_minus_C_est:.2f}, S2-C={val_S2_minus_C_est:.2f}."
                )
                return None

            # exp(-B*deltaX_calib) = (S2-C)/(S1-C). Here deltaX_calib = X2_calib - X1_calib = 1.
            ratio_exp_B_fallback = val_S2_minus_C_est / val_S1_minus_C_est

            if (
                ratio_exp_B_fallback <= 1e-9 or ratio_exp_B_fallback >= 1.0
            ):  # exp(-B) must be in (0,1) for B > 0
                print(
                    f"Warning (Fallback): Ratio for exp(-B) is not suitable ({ratio_exp_B_fallback:.3f}) after C estimation. S1-C={val_S1_minus_C_est:.2f}, S2-C={val_S2_minus_C_est:.2f}"
                )
                return None

            B_fall = -np.log(ratio_exp_B_fallback)
            # B_fall should be > 0 due to previous check.

            exp_term_A_calc = np.exp(-B_fall * X1_calib)
            if abs(exp_term_A_calc) < 1e-9:
                print(
                    f"Warning (Fallback): Exponential term exp(-B*X1_calib) is near zero for A calculation. B={B_fall:.4f}"
                )
                return None

            A_fall = val_S1_minus_C_est / exp_term_A_calc

            print(
                f"Note: Used fallback C estimation for parameters. A={A_fall:.3f}, B={B_fall:.6f}, C={C_est:.3f}"
            )
            return {"A": A_fall, "B": B_fall, "C": C_est}

        except Exception as e:
            print(f"Error calculating parameters: {e}")
            return None

    def calculate_distance(self, sensor_reading, params):
        """Calculate distance from sensor reading using calibration parameters"""
        try:
            A, B, C = params["A"], params["B"], params["C"]
            if sensor_reading <= C:
                return float("inf")  # Invalid reading
            x = -np.log((sensor_reading - C) / A) / B
            return x
        except:
            return float("inf")

    def perform_calibrations(self):
        """Perform reference and various shifted target calibrations using interpolation."""
        ideal_base_points = [1.0, 2.0, 3.0]  # Ideal mm targets

        # 1. Reference calibration using closest actual data to 1,2,3 mm
        print("\\n--- Performing Reference Calibration (Closest Actual Data) ---")
        ref_actual_points_dict = self._find_closest_actual_points(ideal_base_points)

        # The _find_closest_actual_points returns keys as the target, which is what calculate_calibration_parameters expects
        # For example: {1.0: {'actual_distance': 1.002, 'sensor_reading': X}, ...}
        ref_params = self.calculate_calibration_parameters(ref_actual_points_dict)

        if ref_params is None:
            print("Failed to calculate reference_actual_data calibration parameters.")
        else:
            self.calibrations["reference_actual_data"] = {
                "points": ref_actual_points_dict,  # Stores the points used, including their actual distances
                "parameters": ref_params,
                "type": "actual_data_points",
                "shift_mm": 0.0,  # No shift for this type
            }
            print(f"Reference (actual data) calibration points:")
            for target, point_data in ref_actual_points_dict.items():
                print(
                    f"  Target {target:.1f}mm -> Actual {point_data['actual_distance']:.3f}mm, Reading {point_data['sensor_reading']:.0f}"
                )
            print(
                f"  Parameters: A={ref_params['A']:.2f}, B={ref_params['B']:.4f}, C={ref_params['C']:.2f}"
            )

        # 2. Systematic Target Shift Calibrations using Interpolated Sensor Readings
        # Shifts from -500um to +500um in 100um steps
        # np.arange arguments: start, stop (exclusive), step. Add step to stop to make it inclusive.
        shifts_mm = np.round(np.arange(-0.5, 0.5 + 0.1, 0.1), 3)  # in mm

        # Ensure data is sorted by distance for interpolation
        sorted_data = self.data.sort_values("distance")
        xp = sorted_data["distance"].values
        fp = sorted_data["sensor_reading"].values

        print("\\n--- Performing Target Shift Calibrations (Interpolated Data) ---")
        for shift_mm in shifts_mm:
            target_distances_for_interpolation = [
                bp + shift_mm for bp in ideal_base_points
            ]

            # Check if interpolation range is valid
            if not (
                min(target_distances_for_interpolation) >= xp.min()
                and max(target_distances_for_interpolation) <= xp.max()
            ):
                print(
                    f"Shift {shift_mm*1000:+.0f}um ({shift_mm:.3f}mm) - Skipped: Target distances [{min(target_distances_for_interpolation):.2f}, {max(target_distances_for_interpolation):.2f}] outside data range [{xp.min():.2f}, {xp.max():.2f}]."
                )
                continue  # Skip this shift if out of bounds

            interpolated_sensor_readings = np.interp(
                target_distances_for_interpolation, xp, fp
            )

            calib_points_interpolated = {}
            print(
                f"Shift {shift_mm*1000:+.0f}um ({shift_mm:.3f}mm) - Interpolated points for calibration:"
            )
            for i, target_dist_nominal in enumerate(
                ideal_base_points
            ):  # nominal are 1.0, 2.0, 3.0
                # The 'actual_distance' here is the point at which we interpolated a sensor reading
                actual_distance_for_sensor_reading = target_distances_for_interpolation[
                    i
                ]
                sensor_reading = interpolated_sensor_readings[i]
                calib_points_interpolated[target_dist_nominal] = {
                    "actual_distance": actual_distance_for_sensor_reading,
                    "sensor_reading": sensor_reading,
                }
                print(
                    f"  Nominal Target {target_dist_nominal:.1f}mm (actual shifted target {actual_distance_for_sensor_reading:.3f}mm) -> Interpolated Reading {sensor_reading:.0f}"
                )

            calib_params = self.calculate_calibration_parameters(
                calib_points_interpolated
            )
            calib_name = f"target_shift_{shift_mm*1000:+.0f}um"

            if calib_params is not None:
                self.calibrations[calib_name] = {
                    "points": calib_points_interpolated,  # Store points used for this specific calibration
                    "parameters": calib_params,
                    "type": "interpolated_points",
                    "shift_mm": shift_mm,  # Store the shift in mm
                }
                print(
                    f"  Parameters for {calib_name}: A={calib_params['A']:.2f}, B={calib_params['B']:.4f}, C={calib_params['C']:.2f}"
                )
            else:
                print(
                    f"Failed to calculate parameters for {calib_name} with shift {shift_mm:.3f}mm."
                )

        # The old offset calibrations based on self.increment are commented out as requested
        # offsets = [-2, -1, 1, 2]  # steps
        # for offset in offsets:
        #     offset_distance = offset * self.increment
        #     offset_points_targets = [p + offset_distance for p in ideal_base_points]
        #     calib_points_actual = self._find_closest_actual_points(offset_points_targets)
        #     # Create a new dict for calculate_calibration_parameters with ideal_base_points as keys
        #     # and sensor readings from calib_points_actual
        #     points_for_calc = {}
        #     valid_points = True
        #     temp_print_details = []
        #     for i, ideal_target in enumerate(ideal_base_points):
        #         actual_target_for_lookup = offset_points_targets[i]
        #         if actual_target_for_lookup in calib_points_actual:
        #             points_for_calc[ideal_target] = calib_points_actual[actual_target_for_lookup]
        #             temp_print_details.append(
        #                 f"  Target {ideal_target:.1f}mm (orig offset target {actual_target_for_lookup:.3f}mm -> actual {calib_points_actual[actual_target_for_lookup]['actual_distance']:.3f}mm, "
        #                 f"Reading {calib_points_actual[actual_target_for_lookup]['sensor_reading']:.0f})"
        #             )
        #         else: # Should not happen if _find_closest_actual_points works correctly for all targets
        #             print(f"Error: Could not find data for target {actual_target_for_lookup} in offset calibration.")
        #             valid_points = False
        #             break
        #     if not valid_points:
        #         continue

        #     calib_params = self.calculate_calibration_parameters(points_for_calc)
        #     calib_name_old_offset = f"offset_steps_{offset:+d}"
        #     if calib_params is not None:
        #         self.calibrations[calib_name_old_offset] = {
        #             "points": calib_points_actual, # Store the actual points used
        #             "parameters": calib_params,
        #             "type": "actual_data_increment_offset",
        #             "shift_mm": offset_distance # The overall shift in mm
        #         }
        #         print(f"Offset by steps {offset:+d} ({offset_distance:+.3f}mm) calibration points:")
        #         for detail in temp_print_details:
        #             print(detail)
        #         print(f"  Parameters for {calib_name_old_offset}: A={calib_params['A']:.2f}, B={calib_params['B']:.4f}, C={calib_params['C']:.2f}")

    def calculate_errors(self):
        """Calculate distance errors for each calibration"""
        for calib_name, calib_data in self.calibrations.items():
            params = calib_data["parameters"]
            errors_list = []

            for _, row in self.data.iterrows():
                true_distance = row["distance"]
                sensor_reading = row["sensor_reading"]
                calculated_distance = self.calculate_distance(sensor_reading, params)

                if calculated_distance != float("inf") and not np.isnan(
                    calculated_distance
                ):
                    error = calculated_distance - true_distance
                    errors_list.append(error)
                else:
                    errors_list.append(
                        np.nan
                    )  # Append NaN if distance calculation is invalid

            self.errors[calib_name] = np.array(errors_list)

            # Calculate adjusted errors (subtract mean of original error)
            current_original_errors = np.array(errors_list)  # Ensure it's a numpy array

            adjusted_error_values = np.full_like(
                current_original_errors, np.nan, dtype=np.float64
            )

            # Only operate on non-NaN original errors to calculate mean and subtract
            valid_original_errors_mask = ~np.isnan(current_original_errors)
            if np.any(valid_original_errors_mask):
                mean_original_error = np.nanmean(
                    current_original_errors[valid_original_errors_mask]
                )
                adjusted_error_values[valid_original_errors_mask] = (
                    current_original_errors[valid_original_errors_mask]
                    - mean_original_error
                )

            self.adjusted_errors[calib_name] = adjusted_error_values

    def print_results(self):
        """Print calibration parameters and error statistics"""
        print(f"\n=== SENSOR CALIBRATION ANALYSIS - {self.sheet_name} ===\n")
        print(f"Excel file: {self.excel_file}")
        print(f"Sheet: {self.sheet_name}")
        print(f"Data increment detected: {self.increment} mm\n")

        print("Calibration Parameters:")
        print("-" * 70)
        print(f"{'Calibration':<15s} {'A':<15s} {'B':<12s} {'C':<15s}")
        print("-" * 70)
        for name, calib in self.calibrations.items():
            params = calib["parameters"]
            print(
                f"{name:<15s} {params['A']:15.2f} {params['B']:12.6f} {params['C']:15.2f}"
            )

        # Modified Error Statistics section to show Non-Linearity (RMS of Adjusted Error)
        print(f"\nError Statistics (Non-Linearity Focus):")
        print("-" * 80)  # Adjusted width
        print(
            f"{'Calibration':<20s} {'Non-Lin (RMS Adj Err)':<22s} {'Max Orig Err':<18s} {'Mean Orig Err':<18s}"
        )
        print("-" * 80)  # Adjusted width

        for name, errors_arr in self.errors.items():
            # Original Error Stats for Max and Mean
            valid_errors_all = errors_arr[~np.isnan(errors_arr)]
            max_error_all_str = "N/A"
            mean_error_all_str = "N/A"
            if len(valid_errors_all) > 0:
                max_error_all_str = f"{np.max(np.abs(valid_errors_all)):.4f}"
                mean_error_all_str = f"{np.mean(valid_errors_all):.4f}"

            # Adjusted Error Stats for Non-Linearity Score (RMS Adj Err)
            adj_errors_arr_for_name = self.adjusted_errors.get(name)
            rms_adj_error_all_str = "N/A"
            if adj_errors_arr_for_name is not None:
                valid_adj_errors_all = adj_errors_arr_for_name[
                    ~np.isnan(adj_errors_arr_for_name)
                ]
                if len(valid_adj_errors_all) > 0:
                    rms_adj_error_all_str = (
                        f"{np.sqrt(np.mean(valid_adj_errors_all**2)):.4f}"
                    )

            print(
                f"{name:<20s} {rms_adj_error_all_str:<22s} {max_error_all_str:<18s} {mean_error_all_str:<18s} (all data)"
            )

            # --- Statistics for 1-3mm range ---
            range_mask = (self.data["distance"] >= 1.0) & (self.data["distance"] <= 3.0)

            # Original Error Stats for 1-3mm (Max and Mean)
            max_error_1_3_str = "N/A"
            mean_error_1_3_str = "N/A"
            if len(errors_arr) == len(self.data):  # Check alignment
                errors_in_range_mask = range_mask & ~np.isnan(errors_arr)
                errors_in_range = errors_arr[errors_in_range_mask]
                if len(errors_in_range) > 0:
                    max_error_1_3_str = f"{np.max(np.abs(errors_in_range)):.4f}"
                    mean_error_1_3_str = f"{np.mean(errors_in_range):.4f}"

            # Adjusted Error Stats for 1-3mm (Non-Linearity Score)
            rms_adj_error_1_3_str = "N/A"
            if adj_errors_arr_for_name is not None and len(
                adj_errors_arr_for_name
            ) == len(
                self.data
            ):  # Check alignment
                adj_errors_in_range_mask = range_mask & ~np.isnan(
                    adj_errors_arr_for_name
                )
                adj_errors_in_range = adj_errors_arr_for_name[adj_errors_in_range_mask]
                if len(adj_errors_in_range) > 0:
                    rms_adj_error_1_3_str = (
                        f"{np.sqrt(np.mean(adj_errors_in_range**2)):.4f}"
                    )

            if len(errors_arr) == len(self.data):
                print(
                    f"{'':<20s} {rms_adj_error_1_3_str:<22s} {max_error_1_3_str:<18s} {mean_error_1_3_str:<18s} (1-3mm range)"
                )
            else:
                print(
                    f"{'':<20s} {'Error: Mismatch in error array length for 1-3mm stats.':<68s}"
                )

        print(f"\\nAdjusted Error Statistics (Original Error - Mean Original Error):")
        print("-" * 70)
        print(
            f"{'Calibration':<15s} {'RMS Error':<12s} {'Max Error':<12s} {'Mean Error':<12s}"
        )
        print("-" * 70)
        for name, adj_errors_arr in self.adjusted_errors.items():
            # --- Overall Statistics (Adjusted Error) ---
            valid_adj_errors_all = adj_errors_arr[~np.isnan(adj_errors_arr)]
            if len(valid_adj_errors_all) > 0:
                rms_adj_error_all = np.sqrt(np.mean(valid_adj_errors_all**2))
                max_adj_error_all = np.max(np.abs(valid_adj_errors_all))
                mean_adj_error_all = np.mean(
                    valid_adj_errors_all
                )  # Should be close to 0
                print(
                    f"{name:<15s} {rms_adj_error_all:<12.4f} {max_adj_error_all:<12.4f} {mean_adj_error_all:<12.4f} (all data)"
                )
            else:
                print(
                    f"{name:<15s} {'N/A':<12s} {'N/A':<12s} {'N/A':<12s} (all data, no valid errors)"
                )

            # --- Statistics for 1-3mm range (Adjusted Error) ---
            # range_mask is already defined
            if len(adj_errors_arr) == len(self.data):
                adj_errors_in_range_mask = range_mask & ~np.isnan(adj_errors_arr)
                adj_errors_in_range = adj_errors_arr[adj_errors_in_range_mask]

                if len(adj_errors_in_range) > 0:
                    rms_adj_error_1_3 = np.sqrt(np.mean(adj_errors_in_range**2))
                    max_adj_error_1_3 = np.max(np.abs(adj_errors_in_range))
                    mean_adj_error_1_3 = np.mean(adj_errors_in_range)
                    print(
                        f"{'':<15s} {rms_adj_error_1_3:<12.4f} {max_adj_error_1_3:<12.4f} {mean_adj_error_1_3:<12.4f} (1-3mm range)"
                    )
                else:
                    print(
                        f"{'':<15s} {'N/A':<12s} {'N/A':<12s} {'N/A':<12s} (1-3mm range, no valid data)"
                    )
            else:
                print(
                    f"{'':<15s} Error: Mismatch in adjusted error array length for 1-3mm stats."
                )

        # self.find_and_print_optimal_calibration() # Commented out due to AttributeError

    def plot_errors(self):
        """Plot error graphs for each calibration"""
        num_calibrations = len(self.errors)
        if num_calibrations == 0:
            print("No error data to plot.")
            return

        fig, axs = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        # Use a colormap that provides distinct colors for many lines
        num_unique_calibs = len(self.calibrations)
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, max(10, num_unique_calibs)))

        # Plot 1: Original Errors
        ax1 = axs[0]
        for i, (name, errors_arr) in enumerate(self.errors.items()):
            # Ensure errors_arr is aligned with self.data["distance"]
            if len(errors_arr) == len(self.data["distance"]):
                valid_mask = ~np.isnan(errors_arr)
                if np.sum(valid_mask) > 0:  # Check if there's any valid data to plot
                    ax1.plot(
                        self.data["distance"][valid_mask],
                        errors_arr[valid_mask],
                        label=name,
                        marker="o",
                        markersize=3,
                        color=colors[i % len(colors)],
                        alpha=0.7,
                    )
            else:
                print(
                    f"Plotting warning: Mismatch length for original errors of '{name}'"
                )

        ax1.set_ylabel("Original Distance Error (mm)")
        ax1.set_title(f"Original Calibration Error Analysis - {self.sheet_name}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Adjusted Errors (Error - Mean Error)
        ax2 = axs[1]
        for i, (name, adj_errors_arr) in enumerate(self.adjusted_errors.items()):
            if len(adj_errors_arr) == len(self.data["distance"]):
                valid_mask = ~np.isnan(adj_errors_arr)
                if np.sum(valid_mask) > 0:  # Check if there's any valid data to plot
                    ax2.plot(
                        self.data["distance"][valid_mask],
                        adj_errors_arr[valid_mask],
                        label=name,
                        marker="x",
                        markersize=3,
                        color=colors[i % len(colors)],
                        alpha=0.7,
                    )
            else:
                print(
                    f"Plotting warning: Mismatch length for adjusted errors of '{name}'"
                )

        ax2.set_xlabel("True Distance (mm)")
        ax2.set_ylabel("Adjusted Distance Error (mm)\\n(Error - Mean Original Error)")
        ax2.set_title(
            f"Adjusted Calibration Error (Linearity Check) - {self.sheet_name}"
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_results(self, output_dir=".", base_filename="calibration_results"):
        """
        Save comprehensive results to an Excel file with multiple sheets.
        Sheets include:
        1. Calibration Parameters: A, B, C, type, shift for each calibration.
        2. Error Summary: RMS adjusted error (non-linearity), max original error, mean original error
                         for all data and for the 1-3mm range.
        3. Detailed Data: Long format table with true distance, sensor reading, and for each calibration:
                          calculated distance, original error, adjusted error.
        """
        if self.data is None or self.data.empty:
            print(
                f"No data loaded for sheet '{self.sheet_name}'. Skipping result saving."
            )
            return

        if not self.calibrations:
            print(
                f"No calibrations performed for sheet '{self.sheet_name}'. Skipping result saving."
            )
            return

        # Sanitize sheet name for use in filename
        safe_sheet_name = "".join(c if c.isalnum() else "_" for c in self.sheet_name)
        # Ensure output_dir exists - pandas ExcelWriter might not create it.
        # import os # Consider adding import os at the top of the file if not already present
        # os.makedirs(output_dir, exist_ok=True)
        output_file_xlsx = f"{output_dir}/{base_filename}_{safe_sheet_name}.xlsx"

        print(f"Saving results for sheet '{self.sheet_name}' to {output_file_xlsx}...")

        try:
            with pd.ExcelWriter(output_file_xlsx, engine="openpyxl") as writer:
                # Sheet 1: Calibration Parameters
                params_data = []
                for name, calib_info in self.calibrations.items():
                    params = calib_info["parameters"]
                    params_data.append(
                        {
                            "Calibration Name": name,
                            "A": params.get("A"),
                            "B": params.get("B"),
                            "C": params.get("C"),
                            "Type": calib_info.get("type"),
                            "Shift (mm)": calib_info.get("shift_mm"),
                        }
                    )
                df_params = pd.DataFrame(params_data)
                df_params.to_excel(
                    writer, sheet_name="Calibration Parameters", index=False
                )
                print("  Sheet 'Calibration Parameters' saved.")

                # Sheet 2: Error Summary Statistics
                summary_data = []
                range_mask_1_3mm = (self.data["distance"] >= 1.0) & (
                    self.data["distance"] <= 3.0
                )

                for (
                    name
                ) in (
                    self.calibrations.keys()
                ):  # Iterate over names to ensure we attempt all cals
                    orig_errors = self.errors.get(name)
                    adj_errors = self.adjusted_errors.get(name)

                    row_summary = {"Calibration Name": name}

                    if (
                        orig_errors is not None
                        and adj_errors is not None
                        and len(orig_errors) == len(self.data)
                        and len(adj_errors) == len(self.data)
                    ):

                        # All Data
                        valid_orig_all = orig_errors[~np.isnan(orig_errors)]
                        valid_adj_all = adj_errors[~np.isnan(adj_errors)]

                        row_summary["RMS Adj Err (All)"] = (
                            np.sqrt(np.mean(valid_adj_all**2))
                            if len(valid_adj_all) > 0
                            else np.nan
                        )
                        row_summary["Max Orig Err (All)"] = (
                            np.max(np.abs(valid_orig_all))
                            if len(valid_orig_all) > 0
                            else np.nan
                        )
                        row_summary["Mean Orig Err (All)"] = (
                            np.mean(valid_orig_all)
                            if len(valid_orig_all) > 0
                            else np.nan
                        )

                        # 1-3mm Range - Apply range_mask_1_3mm to self.data aligned arrays
                        # Ensure boolean indexing is safe by aligning masks with error arrays of correct length
                        orig_errors_in_range_mask = range_mask_1_3mm & ~np.isnan(
                            orig_errors
                        )
                        adj_errors_in_range_mask = range_mask_1_3mm & ~np.isnan(
                            adj_errors
                        )

                        orig_errors_1_3 = orig_errors[orig_errors_in_range_mask]
                        adj_errors_1_3 = adj_errors[adj_errors_in_range_mask]

                        row_summary["RMS Adj Err (1-3mm)"] = (
                            np.sqrt(np.mean(adj_errors_1_3**2))
                            if len(adj_errors_1_3) > 0
                            else np.nan
                        )
                        row_summary["Max Orig Err (1-3mm)"] = (
                            np.max(np.abs(orig_errors_1_3))
                            if len(orig_errors_1_3) > 0
                            else np.nan
                        )
                        row_summary["Mean Orig Err (1-3mm)"] = (
                            np.mean(orig_errors_1_3)
                            if len(orig_errors_1_3) > 0
                            else np.nan
                        )
                    else:
                        for col in [
                            "RMS Adj Err (All)",
                            "Max Orig Err (All)",
                            "Mean Orig Err (All)",
                            "RMS Adj Err (1-3mm)",
                            "Max Orig Err (1-3mm)",
                            "Mean Orig Err (1-3mm)",
                        ]:
                            row_summary[col] = np.nan
                        if orig_errors is None or adj_errors is None:
                            print(
                                f"    Warning: Error data missing for '{name}' in error summary sheet."
                            )
                        else:  # Length mismatch
                            print(
                                f"    Warning: Mismatch in error array length for '{name}' (orig: {len(orig_errors if orig_errors is not None else [])}, adj: {len(adj_errors if adj_errors is not None else [])}, data: {len(self.data)}) in error summary sheet."
                            )
                    summary_data.append(row_summary)

                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name="Error Summary", index=False)
                print("  Sheet 'Error Summary' saved.")

                # Sheet 3: Detailed Data (Long Format)
                detailed_rows_list = []
                # Prepare base data only once
                base_data_for_detailed = self.data[
                    ["distance", "sensor_reading"]
                ].copy()
                base_data_for_detailed.rename(
                    columns={
                        "distance": "True Distance",
                        "sensor_reading": "Sensor Reading",
                    },
                    inplace=True,
                )

                for calib_name, calib_info_detail in self.calibrations.items():
                    params_detail = calib_info_detail["parameters"]

                    # Create a fresh copy for each calibration to avoid modifying shared data
                    current_calib_df = base_data_for_detailed.copy()
                    current_calib_df["Calibration Name"] = calib_name
                    current_calib_df["Shift (mm)"] = calib_info_detail.get(
                        "shift_mm", np.nan
                    )
                    current_calib_df["Calibration Type"] = calib_info_detail.get(
                        "type", "N/A"
                    )

                    # Calculate distances for this calibration
                    calculated_distances = []
                    for (
                        _,
                        data_row,
                    ) in self.data.iterrows():  # Use self.data to ensure full alignment
                        sensor_val = data_row["sensor_reading"]
                        calculated_distances.append(
                            self.calculate_distance(sensor_val, params_detail)
                        )
                    current_calib_df["Calculated Distance"] = calculated_distances

                    # Get errors, ensuring they are numpy arrays and handle missing ones
                    orig_errors_cal = self.errors.get(calib_name)
                    adj_errors_cal = self.adjusted_errors.get(calib_name)

                    current_calib_df["Original Error"] = (
                        np.array(orig_errors_cal)
                        if orig_errors_cal is not None
                        and len(orig_errors_cal) == len(current_calib_df)
                        else np.nan
                    )
                    current_calib_df["Adjusted Error"] = (
                        np.array(adj_errors_cal)
                        if adj_errors_cal is not None
                        and len(adj_errors_cal) == len(current_calib_df)
                        else np.nan
                    )

                    if orig_errors_cal is None or len(orig_errors_cal) != len(
                        current_calib_df
                    ):
                        print(
                            f"    Warning: Original error data issue for '{calib_name}' in detailed data sheet."
                        )
                    if adj_errors_cal is None or len(adj_errors_cal) != len(
                        current_calib_df
                    ):
                        print(
                            f"    Warning: Adjusted error data issue for '{calib_name}' in detailed data sheet."
                        )

                    detailed_rows_list.append(current_calib_df)

                if detailed_rows_list:
                    df_detailed_all = pd.concat(detailed_rows_list, ignore_index=True)
                    cols_order = [
                        "Calibration Name",
                        "Shift (mm)",
                        "Calibration Type",
                        "True Distance",
                        "Sensor Reading",
                        "Calculated Distance",
                        "Original Error",
                        "Adjusted Error",
                    ]
                    # Filter for columns that actually exist to prevent KeyError
                    existing_cols_order = [
                        col for col in cols_order if col in df_detailed_all.columns
                    ]
                    df_detailed_all = df_detailed_all[existing_cols_order]

                    df_detailed_all.to_excel(
                        writer, sheet_name="Detailed Data", index=False
                    )
                    print("  Sheet 'Detailed Data' saved.")
                else:
                    print("  No detailed data to save.")

            print(f"Successfully saved results to {output_file_xlsx}")
            print(
                "Note: The 'openpyxl' library is required to write Excel files. If not installed, run: pip install openpyxl"
            )

        except ImportError:
            print(
                "Error: The 'openpyxl' library is required to write Excel files. Please install it using 'pip install openpyxl'."
            )
        except Exception as e:
            print(f"Error saving results to Excel: {e}")


# ... existing code ...
# Make sure this method replaces the old save_results method.
# The class definition and other methods remain the same unless specified.

if __name__ == "__main__":
    # --- CONFIGURATION ---
    excel_file_path = "C:\\\\\\\\Users\\\\\\\\geller\\\\\\\\OneDrive - HP Inc\\\\\\\\data\\\\\\\\ROS\\\\\\\\ROS calibration for series 4\\\\\\\\manual calibration of ROS agasint series 4 drum\\\\\\\\9230004 manual measurements\\\\\\\\channels 1 and 2 summary.xlsx"
    sheets_to_analyze = ["channel 1 all", "channel 2 all"]  # Updated list of sheets
    output_directory = "analysis_results"  # Directory to save the Excel results
    base_output_filename_prefix = "calibration_analysis"  # Prefix for output files

    # --- CREATE OUTPUT DIRECTORY IF IT DOESN'T EXIST ---
    import os

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # --- EXECUTION FOR EACH SHEET ---
    for sheet_name in sheets_to_analyze:
        print(
            f"\\n\\n--- Starting analysis for: {excel_file_path} - Sheet: {sheet_name} ---"
        )

        # Create an analyzer instance
        analyzer = SensorCalibrationAnalyzer(excel_file_path, sheet_name)

        if analyzer.data is not None and not analyzer.data.empty:
            # Perform calibrations
            analyzer.perform_calibrations()

            # Calculate errors
            analyzer.calculate_errors()

            # Print results to console
            analyzer.print_results()

            # Plot errors
            # Note: plt.show() is blocking. Plots for each sheet will appear sequentially.
            print(
                f"Displaying plots for sheet '{sheet_name}'. Close the plot window to continue..."
            )
            analyzer.plot_errors()

            # Save results to Excel
            # Sanitize sheet name for use in filename
            safe_sheet_name_for_file = "".join(
                c if c.isalnum() else "_" for c in sheet_name
            )
            sheet_specific_base_filename = (
                f"{base_output_filename_prefix}_{safe_sheet_name_for_file}"
            )
            analyzer.save_results(
                output_dir=output_directory, base_filename=sheet_specific_base_filename
            )

            print(f"\\\\nAnalysis complete for sheet '{sheet_name}'.")
            print(f"Plots were displayed.")
            print(
                f"Results saved to an Excel file in '{output_directory}' starting with '{sheet_specific_base_filename}'."
            )
        else:
            print(
                f"Could not proceed with analysis for sheet '{sheet_name}' due to data loading issues."
            )

    print("\\\\n\\nAll specified sheets processed.")
