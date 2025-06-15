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
        """Perform calibrations using a unified method: actual data when available, interpolated otherwise."""
        ideal_base_points = [1.0, 2.0, 3.0]  # Ideal mm targets

        # Shifts from -500um to +500um in 100um steps, including 0
        shifts_mm = np.round(np.arange(-0.5, 0.5 + 0.1, 0.1), 3)  # in mm

        # Ensure data is sorted by distance for interpolation
        sorted_data = self.data.sort_values("distance")
        xp = sorted_data["distance"].values
        fp = sorted_data["sensor_reading"].values

        print(
            "\\n--- Performing Unified Calibrations (Actual Data Preferred, Interpolated When Needed) ---"
        )

        for shift_mm in shifts_mm:
            target_distances = [bp + shift_mm for bp in ideal_base_points]

            # Check if interpolation range is valid
            if not (
                min(target_distances) >= xp.min() and max(target_distances) <= xp.max()
            ):
                print(
                    f"Shift {shift_mm*1000:+.0f}um ({shift_mm:.3f}mm) - Skipped: Target distances [{min(target_distances):.2f}, {max(target_distances):.2f}] outside data range [{xp.min():.2f}, {xp.max():.2f}]."
                )
                continue

            calib_points = {}
            print(
                f"Shift {shift_mm*1000:+.0f}um ({shift_mm:.3f}mm) - Calibration points:"
            )

            for i, (nominal_target, actual_target) in enumerate(
                zip(ideal_base_points, target_distances)
            ):
                # Check if we have exact actual data for this target distance
                exact_match_mask = np.abs(self.data["distance"] - actual_target) < 1e-6

                if np.any(exact_match_mask):
                    # Use actual data point
                    exact_idx = np.where(exact_match_mask)[0][0]
                    actual_distance = self.data["distance"].iloc[exact_idx]
                    sensor_reading = self.data["sensor_reading"].iloc[exact_idx]
                    data_type = "ACTUAL"
                else:
                    # Use interpolated data
                    sensor_reading = np.interp(actual_target, xp, fp)
                    actual_distance = actual_target
                    data_type = "INTERP"

                calib_points[nominal_target] = {
                    "actual_distance": actual_distance,
                    "sensor_reading": sensor_reading,
                }

                print(
                    f"  Target {nominal_target:.1f}mm -> {actual_distance:.3f}mm, Reading {sensor_reading:.0f} [{data_type}]"
                )

            # Calculate calibration parameters
            calib_params = self.calculate_calibration_parameters(calib_points)

            if shift_mm == 0.0:
                calib_name = "reference_0um"
            else:
                calib_name = f"shift_{shift_mm*1000:+.0f}um"

            if calib_params is not None:
                self.calibrations[calib_name] = {
                    "points": calib_points,
                    "parameters": calib_params,
                    "type": "unified_method",
                    "shift_mm": shift_mm,
                }
                print(
                    f"  Parameters: A={calib_params['A']:.2f}, B={calib_params['B']:.4f}, C={calib_params['C']:.2f}"
                )
            else:
                print(f"Failed to calculate parameters for {calib_name}")

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
        Save results organized by metrics for better comparability.
        Creates separate sheets for each metric, showing how different position shifts affect that parameter.
        Each sheet contains data for all calibrations to enable easy comparison.
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
        output_file_xlsx = f"{output_dir}/{base_filename}_{safe_sheet_name}.xlsx"

        print(f"Saving results for sheet '{self.sheet_name}' to {output_file_xlsx}...")

        try:
            with pd.ExcelWriter(output_file_xlsx, engine="openpyxl") as writer:
                # Prepare data structures for metric-based organization
                range_mask_1_3mm = (self.data["distance"] >= 1.0) & (
                    self.data["distance"] <= 3.0
                )

                # Extract calibration info for organizing
                calib_data = []
                for name, calib_info in self.calibrations.items():
                    params = calib_info["parameters"]
                    orig_errors = self.errors.get(name)
                    adj_errors = self.adjusted_errors.get(name)

                    row_data = {
                        "Calibration_Name": name,
                        "Shift_mm": calib_info.get("shift_mm", np.nan),
                        "Type": calib_info.get("type", "N/A"),
                        "A": params.get("A", np.nan),
                        "B": params.get("B", np.nan),
                        "C": params.get("C", np.nan),
                    }

                    # Calculate error metrics
                    if (
                        orig_errors is not None
                        and adj_errors is not None
                        and len(orig_errors) == len(self.data)
                        and len(adj_errors) == len(self.data)
                    ):

                        # All data metrics
                        valid_orig_all = orig_errors[~np.isnan(orig_errors)]
                        valid_adj_all = adj_errors[~np.isnan(adj_errors)]

                        row_data["RMS_Adj_Err_All"] = (
                            np.sqrt(np.mean(valid_adj_all**2))
                            if len(valid_adj_all) > 0
                            else np.nan
                        )
                        row_data["Max_Orig_Err_All"] = (
                            np.max(np.abs(valid_orig_all))
                            if len(valid_orig_all) > 0
                            else np.nan
                        )
                        row_data["Mean_Orig_Err_All"] = (
                            np.mean(valid_orig_all)
                            if len(valid_orig_all) > 0
                            else np.nan
                        )

                        # 1-3mm range metrics
                        orig_errors_in_range_mask = range_mask_1_3mm & ~np.isnan(
                            orig_errors
                        )
                        adj_errors_in_range_mask = range_mask_1_3mm & ~np.isnan(
                            adj_errors
                        )

                        orig_errors_1_3 = orig_errors[orig_errors_in_range_mask]
                        adj_errors_1_3 = adj_errors[adj_errors_in_range_mask]

                        row_data["RMS_Adj_Err_1_3mm"] = (
                            np.sqrt(np.mean(adj_errors_1_3**2))
                            if len(adj_errors_1_3) > 0
                            else np.nan
                        )
                        row_data["Max_Orig_Err_1_3mm"] = (
                            np.max(np.abs(orig_errors_1_3))
                            if len(orig_errors_1_3) > 0
                            else np.nan
                        )
                        row_data["Mean_Orig_Err_1_3mm"] = (
                            np.mean(orig_errors_1_3)
                            if len(orig_errors_1_3) > 0
                            else np.nan
                        )
                    else:
                        # Fill with NaN if data is invalid
                        for col in [
                            "RMS_Adj_Err_All",
                            "Max_Orig_Err_All",
                            "Mean_Orig_Err_All",
                            "RMS_Adj_Err_1_3mm",
                            "Max_Orig_Err_1_3mm",
                            "Mean_Orig_Err_1_3mm",
                        ]:
                            row_data[col] = np.nan

                    calib_data.append(row_data)

                df_all_data = pd.DataFrame(calib_data)

                # Sort by shift for better readability
                df_all_data = df_all_data.sort_values("Shift_mm").reset_index(drop=True)

                # Sheet 1: Calibration Parameters (A, B, C vs Position Shift)
                params_sheet = df_all_data[["Shift_mm", "Type", "A", "B", "C"]].copy()
                params_sheet.to_excel(
                    writer, sheet_name="Parameters_vs_Shift", index=False
                )
                print("  Sheet 'Parameters_vs_Shift' saved.")

                # Sheet 2: Non-Linearity Metrics (RMS Adjusted Errors)
                nonlin_sheet = df_all_data[
                    ["Shift_mm", "Type", "RMS_Adj_Err_All", "RMS_Adj_Err_1_3mm"]
                ].copy()
                nonlin_sheet.rename(
                    columns={
                        "RMS_Adj_Err_All": "RMS_Adjusted_Error_All_Data",
                        "RMS_Adj_Err_1_3mm": "RMS_Adjusted_Error_1_3mm_Range",
                    },
                    inplace=True,
                )
                nonlin_sheet.to_excel(
                    writer, sheet_name="NonLinearity_vs_Shift", index=False
                )
                print("  Sheet 'NonLinearity_vs_Shift' saved.")

                # Sheet 3: Maximum Original Errors
                max_err_sheet = df_all_data[
                    ["Shift_mm", "Type", "Max_Orig_Err_All", "Max_Orig_Err_1_3mm"]
                ].copy()
                max_err_sheet.rename(
                    columns={
                        "Max_Orig_Err_All": "Max_Original_Error_All_Data",
                        "Max_Orig_Err_1_3mm": "Max_Original_Error_1_3mm_Range",
                    },
                    inplace=True,
                )
                max_err_sheet.to_excel(
                    writer, sheet_name="MaxError_vs_Shift", index=False
                )
                print("  Sheet 'MaxError_vs_Shift' saved.")

                # Sheet 4: Mean Original Errors (Bias)
                mean_err_sheet = df_all_data[
                    ["Shift_mm", "Type", "Mean_Orig_Err_All", "Mean_Orig_Err_1_3mm"]
                ].copy()
                mean_err_sheet.rename(
                    columns={
                        "Mean_Orig_Err_All": "Mean_Original_Error_All_Data",
                        "Mean_Orig_Err_1_3mm": "Mean_Original_Error_1_3mm_Range",
                    },
                    inplace=True,
                )
                mean_err_sheet.to_excel(
                    writer, sheet_name="MeanError_vs_Shift", index=False
                )
                print("  Sheet 'MeanError_vs_Shift' saved.")
                # Sheet 5: Summary Overview
                summary_sheet = df_all_data[
                    [
                        "Calibration_Name",
                        "Shift_mm",
                        "Type",
                        "RMS_Adj_Err_All",
                        "Max_Orig_Err_All",
                        "Mean_Orig_Err_All",
                    ]
                ].copy()
                summary_sheet.rename(
                    columns={
                        "Calibration_Name": "Calibration_Name",
                        "RMS_Adj_Err_All": "NonLinearity_RMS",
                        "Max_Orig_Err_All": "Max_Error",
                        "Mean_Orig_Err_All": "Bias",
                    },
                    inplace=True,
                )
                summary_sheet.to_excel(
                    writer, sheet_name="Summary_Overview", index=False
                )
                print("  Sheet 'Summary_Overview' saved.")

                # Sheet 6: Sensor Distance Readings vs True Distance for Each Calibration
                sensor_readings_sheet_data = {
                    "True_Distance_mm": self.data["distance"].values
                }

                for name, calib_info in self.calibrations.items():
                    params = calib_info["parameters"]
                    calculated_distances = []

                    for _, row in self.data.iterrows():
                        sensor_reading = row["sensor_reading"]
                        calc_dist = self.calculate_distance(sensor_reading, params)
                        if calc_dist != float("inf") and not np.isnan(calc_dist):
                            calculated_distances.append(calc_dist)
                        else:
                            calculated_distances.append(np.nan)

                    shift_info = (
                        f"_shift_{calib_info.get('shift_mm', 0)*1000:+.0f}um"
                        if calib_info.get("shift_mm") is not None
                        else ""
                    )
                    col_name = f"Calc_Dist_{name}{shift_info}"
                    sensor_readings_sheet_data[col_name] = calculated_distances

                sensor_readings_df = pd.DataFrame(sensor_readings_sheet_data)
                sensor_readings_df.to_excel(
                    writer, sheet_name="Calculated_Distances_vs_True", index=False
                )
                print("  Sheet 'Calculated_Distances_vs_True' saved.")

                # Sheet 7: Distance Errors vs True Distance for Each Calibration
                distance_errors_sheet_data = {
                    "True_Distance_mm": self.data["distance"].values
                }

                for name, calib_info in self.calibrations.items():
                    errors_arr = self.errors.get(name)
                    if errors_arr is not None and len(errors_arr) == len(self.data):
                        shift_info = (
                            f"_shift_{calib_info.get('shift_mm', 0)*1000:+.0f}um"
                            if calib_info.get("shift_mm") is not None
                            else ""
                        )
                        col_name = f"Error_{name}{shift_info}"
                        distance_errors_sheet_data[col_name] = errors_arr
                    else:
                        # Fill with NaN if no valid error data
                        shift_info = (
                            f"_shift_{calib_info.get('shift_mm', 0)*1000:+.0f}um"
                            if calib_info.get("shift_mm") is not None
                            else ""
                        )
                        col_name = f"Error_{name}{shift_info}"
                        distance_errors_sheet_data[col_name] = [np.nan] * len(self.data)

                distance_errors_df = pd.DataFrame(distance_errors_sheet_data)
                distance_errors_df.to_excel(
                    writer, sheet_name="Distance_Errors_vs_True", index=False
                )
                print("  Sheet 'Distance_Errors_vs_True' saved.")

            print(f"Successfully saved results to {output_file_xlsx}")
            print(
                "Data organized by metrics for easy comparison across position shifts."
            )

        except ImportError:
            print(
                "Error: The 'openpyxl' library is required to write Excel files. Please install it using 'pip install openpyxl'."
            )
        except Exception as e:
            print(f"Error saving results to Excel: {e}")

    def plot_raw_and_interpolated_data(self):
        """Plot raw data and show interpolated points before analysis begins"""
        if self.data is None or self.data.empty:
            print("No data available for plotting.")
            return

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        # Plot raw data
        ax.scatter(
            self.data["distance"],
            self.data["sensor_reading"],
            c="blue",
            s=30,
            alpha=0.6,
            label="Raw Data Points",
            zorder=3,
        )

        # Create interpolation curve for visualization
        distance_range = np.linspace(
            self.data["distance"].min(), self.data["distance"].max(), 500
        )

        # Ensure data is sorted for interpolation
        sorted_data = self.data.sort_values("distance")
        xp = sorted_data["distance"].values
        fp = sorted_data["sensor_reading"].values

        interpolated_curve = np.interp(distance_range, xp, fp)
        ax.plot(
            distance_range,
            interpolated_curve,
            "r-",
            linewidth=2,
            alpha=0.7,
            label="Interpolation Curve",
            zorder=2,
        )

        # Show calibration target points for different shifts
        ideal_base_points = [1.0, 2.0, 3.0]
        shifts_to_show = [-0.3, 0.0, 0.3]  # Show a few example shifts
        colors = ["green", "orange", "purple"]

        for shift_mm, color in zip(shifts_to_show, colors):
            target_distances = [bp + shift_mm for bp in ideal_base_points]

            # Check if targets are within data range
            if min(target_distances) >= xp.min() and max(target_distances) <= xp.max():
                interp_readings = np.interp(target_distances, xp, fp)

                ax.scatter(
                    target_distances,
                    interp_readings,
                    c=color,
                    s=80,
                    marker="s",
                    alpha=0.8,
                    label=f"Calib Points (shift {shift_mm*1000:+.0f}μm)",
                    zorder=4,
                    edgecolors="black",
                    linewidth=1,
                )

                # Add annotations for calibration points
                for dist, reading in zip(target_distances, interp_readings):
                    ax.annotate(
                        f"{dist:.1f}mm",
                        (dist, reading),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.8,
                    )

        ax.set_xlabel("True Distance (mm)", fontsize=12)
        ax.set_ylabel("Sensor Reading", fontsize=12)
        ax.set_title(
            f"Raw Data and Interpolation Overview - {self.sheet_name}", fontsize=14
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Add data statistics text box
        stats_text = f"Data Points: {len(self.data)}\\n"
        stats_text += f'Distance Range: {self.data["distance"].min():.2f} - {self.data["distance"].max():.2f} mm\\n'
        stats_text += f'Sensor Range: {self.data["sensor_reading"].min():.0f} - {self.data["sensor_reading"].max():.0f}\\n'
        stats_text += f"Increment: {self.increment:.3f} mm"

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()

        print(f"Raw data visualization complete for '{self.sheet_name}'.")
        print(
            f"Green, orange, and purple squares show example calibration points for different position shifts."
        )

    def plot_nonlinearity_vs_shift(self):
        """Plot non-linearity (RMS adjusted error) vs calibration misposition"""
        if not self.calibrations:
            print("No calibrations available for non-linearity plot.")
            return

        # Prepare data for plotting
        shifts_mm = []
        nonlinearity_all = []
        nonlinearity_1_3mm = []
        calib_names = []

        range_mask_1_3mm = (self.data["distance"] >= 1.0) & (
            self.data["distance"] <= 3.0
        )

        for name, calib_info in self.calibrations.items():
            shift_mm = calib_info.get("shift_mm")

            # Include all calibrations with shift information
            if shift_mm is None:
                continue

            adj_errors = self.adjusted_errors.get(name)
            if adj_errors is not None and len(adj_errors) == len(self.data):
                # All data non-linearity
                valid_adj_all = adj_errors[~np.isnan(adj_errors)]
                if len(valid_adj_all) > 0:
                    rms_adj_all = np.sqrt(np.mean(valid_adj_all**2))
                else:
                    rms_adj_all = np.nan

                # 1-3mm range non-linearity
                adj_errors_in_range_mask = range_mask_1_3mm & ~np.isnan(adj_errors)
                adj_errors_1_3 = adj_errors[adj_errors_in_range_mask]
                if len(adj_errors_1_3) > 0:
                    rms_adj_1_3 = np.sqrt(np.mean(adj_errors_1_3**2))
                else:
                    rms_adj_1_3 = np.nan

                shifts_mm.append(shift_mm * 1000)  # Convert to micrometers
                nonlinearity_all.append(rms_adj_all)
                nonlinearity_1_3mm.append(rms_adj_1_3)
                calib_names.append(name)

        if not shifts_mm:
            print("No valid shift data available for non-linearity plot.")
            return

        # Sort data by shift for proper line plotting
        sorted_indices = np.argsort(shifts_mm)
        shifts_mm = [shifts_mm[i] for i in sorted_indices]
        nonlinearity_all = [nonlinearity_all[i] for i in sorted_indices]
        nonlinearity_1_3mm = [nonlinearity_1_3mm[i] for i in sorted_indices]
        calib_names = [calib_names[i] for i in sorted_indices]

        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Plot both datasets
        ax.plot(
            shifts_mm,
            nonlinearity_all,
            "o-",
            label="All Data Range",
            marker="o",
            markersize=8,
            linewidth=2,
            alpha=0.8,
        )
        ax.plot(
            shifts_mm,
            nonlinearity_1_3mm,
            "s-",
            label="1-3mm Range",
            marker="s",
            markersize=8,
            linewidth=2,
            alpha=0.8,
        )

        ax.set_xlabel("Calibration Misposition (μm)", fontsize=12)
        ax.set_ylabel("Non-Linearity (RMS Adjusted Error, mm)", fontsize=12)
        ax.set_title(
            f"Non-Linearity vs Calibration Misposition - {self.sheet_name}", fontsize=14
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        # Add data point labels for better readability
        for i, (shift, nl_all, nl_1_3) in enumerate(
            zip(shifts_mm, nonlinearity_all, nonlinearity_1_3mm)
        ):
            if not np.isnan(nl_all):
                ax.annotate(
                    f"{nl_all:.3f}",
                    (shift, nl_all),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=9,
                )
            if not np.isnan(nl_1_3):
                ax.annotate(
                    f"{nl_1_3:.3f}",
                    (shift, nl_1_3),
                    textcoords="offset points",
                    xytext=(0, -15),
                    ha="center",
                    fontsize=9,
                )

        # Improve layout
        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print(f"\n=== Non-Linearity vs Shift Summary - {self.sheet_name} ===")
        print(f"{'Shift (μm)':<12} {'All Data':<12} {'1-3mm Range':<12}")
        print("-" * 40)
        for shift, nl_all, nl_1_3 in zip(
            shifts_mm, nonlinearity_all, nonlinearity_1_3mm
        ):
            print(f"{shift:+8.0f}     {nl_all:8.4f}     {nl_1_3:8.4f}")


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
            # Plot raw data and interpolation overview first
            print(
                f"Displaying raw data overview for sheet '{sheet_name}'. Close the plot window to continue..."
            )
            analyzer.plot_raw_and_interpolated_data()

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

            # Plot non-linearity vs shift
            print(
                f"Displaying non-linearity vs shift plot for sheet '{sheet_name}'. Close the plot window to continue..."
            )
            analyzer.plot_nonlinearity_vs_shift()

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

            print(f"\\nAnalysis complete for sheet '{sheet_name}'.")
            print(f"Plots were displayed.")
            print(
                f"Results saved to an Excel file in '{output_directory}' starting with '{sheet_specific_base_filename}'."
            )
        else:
            print(
                f"Could not proceed with analysis for sheet '{sheet_name}' due to data loading issues."
            )

    print("\\\\n\\nAll specified sheets processed.")
