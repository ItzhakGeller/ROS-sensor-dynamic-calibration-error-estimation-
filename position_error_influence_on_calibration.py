from scipy.optimize import fsolve
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


class SensorCalibrationAnalyzer:
    def __init__(self, excel_file_path, sheet_name, unit_column=None):
        self.excel_file = excel_file_path
        self.sheet_name = sheet_name
        self.unit_column = unit_column  # New attribute for unit column
        self.data = self.load_excel_data()
        if self.data is not None:
            self.increment = self.detect_increment()
            self.calibrations = {}
            self.errors = {}
            self.adjusted_errors = {}
        else:
            print("Failed to load data.")

    def load_excel_data(self):
        """Load Excel data - supports both raw unit columns and processed data"""
        if self.unit_column:
            return self.load_data_with_unit_column()
        else:
            return self.load_processed_data()

    def load_data_with_unit_column(self):
        """Load Excel data from 'all raw' sheet, using column A for distance and specified column for sensor data"""
        try:
            # Read the Excel file
            df_raw = pd.read_excel(self.excel_file, sheet_name=self.sheet_name)

            print(f"Raw data shape: {df_raw.shape}")
            print("First few rows:")
            print(df_raw.head())

            # Get sensor name from row 2 (index 1) of the specified column
            col_index = ord(self.unit_column.upper()) - ord(
                "A"
            )  # Convert A,B,C,D,E to 0,1,2,3,4

            if col_index < df_raw.shape[1] and len(df_raw) > 1:
                # Read sensor name from row 2 (index 1) - the actual header row
                sensor_name = df_raw.iloc[
                    1, col_index
                ]  # Row 2 contains sensor names like "43220065 ROS1"
                print(
                    f"Sensor name from header (row 2, column {self.unit_column}): {sensor_name}"
                )

                # Clean the sensor name - remove any NaN or numeric artifacts
                if (
                    pd.isna(sensor_name)
                    or str(sensor_name).replace(".", "").replace(",", "").isdigit()
                ):
                    # If we got a number instead of a name, try row 1
                    if len(df_raw) > 0:
                        sensor_name = df_raw.iloc[0, col_index]
                        print(f"Trying row 1 instead: {sensor_name}")

                    # If still not good, use default
                    if (
                        pd.isna(sensor_name)
                        or str(sensor_name).replace(".", "").replace(",", "").isdigit()
                    ):
                        sensor_name = f"Sensor_{self.unit_column}"
                        print(f"Using default name: {sensor_name}")

                self.sensor_name = str(sensor_name).strip()
            else:
                self.sensor_name = f"Sensor_{self.unit_column}"
                print(f"Could not read sensor name, using default: {self.sensor_name}")

            # Start reading data from row 3 (index 2) onwards - skip header rows
            data_start_row = 2

            # Extract distance (column A) and sensor data (specified column B, C, D, or E)
            distance_data = df_raw.iloc[data_start_row:, 0]  # Column A - distance
            sensor_data = df_raw.iloc[
                data_start_row:, col_index
            ]  # Specified column - sensor reading

            print(f"Data start row: {data_start_row}")
            print(f"Distance data preview: {distance_data.head().values}")
            print(f"Sensor data preview: {sensor_data.head().values}")

            # Convert to numeric and remove NaN values
            distance_numeric = pd.to_numeric(distance_data, errors="coerce")
            sensor_numeric = pd.to_numeric(sensor_data, errors="coerce")

            # Create DataFrame and remove rows with NaN values
            df = pd.DataFrame(
                {"distance": distance_numeric, "sensor_reading": sensor_numeric}
            ).dropna()

            # Filter to distance <= 4 as specified in the original request
            df = df[df["distance"] <= 4].copy()

            print(f"Loaded {len(df)} data points for {self.sensor_name}")
            if len(df) > 0:
                print(
                    f"Distance range: {df['distance'].min():.3f} - {df['distance'].max():.3f} mm"
                )
                print(
                    f"Sensor range: {df['sensor_reading'].min():.0f} - {df['sensor_reading'].max():.0f}"
                )

            return df

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def load_processed_data(self):
        """Load Excel data from specified sheet, columns A,B starting from row 2"""
        try:
            df_raw = pd.read_excel(
                self.excel_file, sheet_name=self.sheet_name, header=None
            )

            # Extract data starting from row 2 (index 1)
            df_data = df_raw.iloc[1:, [0, 1]].copy()
            df_data.columns = ["distance", "sensor_reading"]

            # Convert to numeric
            df_data["distance"] = pd.to_numeric(df_data["distance"], errors="coerce")
            df_data["sensor_reading"] = pd.to_numeric(
                df_data["sensor_reading"], errors="coerce"
            )

            # Remove NaN values
            df_clean = df_data.dropna().reset_index(drop=True)

            print(f"Loaded {len(df_clean)} data points from processed data")
            return df_clean

        except Exception as e:
            print(f"Error loading processed data: {e}")
            return None

    def load_data(self):
        """Load and process data from Excel sheet with improved structure detection"""
        try:
            df_raw = pd.read_excel(
                self.excel_file, sheet_name=self.sheet_name, header=None
            )
            print(f"Raw data shape: {df_raw.shape}")

            # Detect data start row and columns
            data_start_row = self._detect_data_start_row(df_raw)
            distance_col, sensor_cols = self._detect_data_columns(
                df_raw, data_start_row
            )

            print(f"Detected data start row: {data_start_row}")
            print(f"Detected distance column: {distance_col}")
            print(f"Detected sensor columns: {sensor_cols}")

            # Extract data
            if data_start_row < len(df_raw) and len(sensor_cols) > 0:
                distance_data = df_raw.iloc[data_start_row:, distance_col]
                sensor_data = df_raw.iloc[data_start_row:, sensor_cols[0]]

                # Create DataFrame
                df = pd.DataFrame(
                    {
                        "distance": pd.to_numeric(distance_data, errors="coerce"),
                        "sensor_reading": pd.to_numeric(sensor_data, errors="coerce"),
                    }
                )

                # Clean data
                original_count = len(df)
                df = df.dropna().reset_index(drop=True)
                print(
                    f"Data cleaning: {original_count} -> {len(df)} points ({original_count - len(df)} removed)"
                )

                return df
            else:
                print("Could not detect valid data structure")
                return None

        except Exception as e:
            print(f"Error in load_data: {e}")
            return None

    def _detect_data_start_row(self, df_raw):
        """Detect which row contains the start of numeric data"""
        for row_idx in range(min(10, len(df_raw))):
            try:
                first_cell = pd.to_numeric(df_raw.iloc[row_idx, 0], errors="coerce")
                if not pd.isna(first_cell):
                    return row_idx
            except:
                continue

        return 0  # Default to first row

    def _detect_data_columns(self, df_raw, data_start_row):
        """Detect which columns contain distance and sensor data"""
        if data_start_row >= len(df_raw):
            return 0, [1]

        data_portion = df_raw.iloc[
            data_start_row : data_start_row + 50
        ].copy()  # Look at first 50 data rows

        distance_candidates = []
        sensor_candidates = []

        for col_idx in range(min(15, data_portion.shape[1])):
            numeric_data = pd.to_numeric(data_portion.iloc[:, col_idx], errors="coerce")
            valid_count = numeric_data.notna().sum()

            if valid_count > 10:  # Need at least 10 valid points
                unique_ratio = len(numeric_data.dropna().unique()) / valid_count
                distance_candidates.append((col_idx, unique_ratio, valid_count))
                sensor_candidates.append((col_idx, valid_count))

        # Choose best distance column (highest uniqueness)
        distance_col = 0  # Default to column A
        if distance_candidates:
            distance_col = max(distance_candidates, key=lambda x: x[1])[0]

        # Choose sensor columns (prefer those with more data)
        sensor_cols = [1]  # Default to column B
        if sensor_candidates:
            sensor_cols = [max(sensor_candidates, key=lambda x: x[1])[0]]

        return distance_col, sensor_cols

    def detect_increment(self):
        """Detect data increment from the dataset"""
        if self.data is None or len(self.data) < 2:
            return 0.1  # Default increment

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
            sensor_reading = self.data["sensor_reading"].iloc[closest_idx]
            calibration_points[target] = {
                "actual_distance": actual_distance,
                "sensor_reading": sensor_reading,
            }
        return calibration_points

    def calculate_calibration_parameters(self, points, verbose=True):
        """Calculate A, B, C parameters using the correct exponential decay model equations.

        Model: Sensor = A + C * exp(-B * Distance)
        Where B > 0 for decreasing ROS sensors
          DETAILED COEFFICIENT CALCULATION PROCESS:
        ==========================================

        1. INPUT: Three sensor readings (S1, S2, S3) at distances (X1=1.0, X2=2.0, X3=3.0) mm

        2. MATHEMATICAL MODEL:
           - S1 = A + C * exp(-B * X1)  where X1 = 1.0 mm
           - S2 = A + C * exp(-B * X2)  where X2 = 2.0 mm
           - S3 = A + C * exp(-B * X3)  where X3 = 3.0 mm

        3. COEFFICIENT B CALCULATION (CORRECTED FORMULA):
           B = ln(((S1-S3) ± √((S1-S3)²-4*(S2-S3)*(S1-S2))) / (2*(S2-S3)))

           Steps:
           a) Calculate terms: (S1-S3), (S2-S3), (S1-S2)
           b) Calculate discriminant: (S1-S3)² - 4*(S2-S3)*(S1-S2)
           c) Calculate numerator: (S1-S3) ± √(discriminant)
           d) Calculate denominator: 2*(S2-S3)
           e) Calculate ln(numerator/denominator)
           f) Choose the positive solution (B > 0 for ROS sensors)

        4. COEFFICIENT A CALCULATION:
           A = (S1-S3) / (exp(-B*X1) - exp(-B*X3))

        5. COEFFICIENT C CALCULATION:
           C = S2 - A*exp(-B*X2)

        6. VERIFICATION:
           Check if calculated coefficients reproduce original sensor readings
        """
        try:
            S1, S2, S3 = points
            X1, X2, X3 = 1.0, 2.0, 3.0  # Fixed calibration distances

            if verbose:
                print(
                    f"DEBUG: Calibration points - S1={S1:.1f}, S2={S2:.1f}, S3={S3:.1f}"
                )
                print(f"DEBUG: Calibration distances - X1={X1}, X2={X2}, X3={X3}")

            # Check sensor behavior - for ROS, readings should decrease with distance
            if S1 < S2 or S2 < S3:
                if verbose:
                    print(
                        f"WARNING: Sensor readings should decrease with distance for ROS sensors!"
                    )
                    print(
                        f"S1={S1:.1f} at {X1}mm, S2={S2:.1f} at {X2}mm, S3={S3:.1f} at {X3}mm"
                    )

            # Use the correct equations from the attachment:
            # B = ln(((S1-S3) ± SQRT((S1-S3)²-4*(S2-S3)*(S1-S2)))/(2*(S2-S3)))
            term1 = S1 - S3
            term2 = S2 - S3
            term3 = S1 - S2

            if verbose:
                print(
                    f"DEBUG: Terms - (S1-S3)={term1:.1f}, (S2-S3)={term2:.1f}, (S1-S2)={term3:.1f}"
                )

            # Calculate discriminant
            discriminant = term1**2 - 4 * term2 * term3
            if verbose:
                print(
                    f"DEBUG: Discriminant = {term1:.1f}² - 4*{term2:.1f}*{term3:.1f} = {discriminant:.1f}"
                )

            if discriminant < 0:
                print(
                    f"ERROR: Negative discriminant {discriminant:.1f} - cannot solve for B"
                )
                return {"A": 0, "B": 0, "C": 0}

            sqrt_discriminant = np.sqrt(discriminant)

            # Apply the correct formula: B = ln(((S1-S3) ± SQRT((S1-S3)²-4*(S2-S3)*(S1-S2)))/(2*(S2-S3)))
            denominator = 2 * term2  # 2*(S2-S3)
            if abs(denominator) < 1e-10:
                print(f"ERROR: Denominator too small: {denominator}")
                return {
                    "A": 0,
                    "B": 0,
                    "C": 0,
                }  # Calculate the two possible arguments for the natural logarithm
            numerator_plus = term1 + sqrt_discriminant  # (S1-S3) + SQRT(...)
            numerator_minus = term1 - sqrt_discriminant  # (S1-S3) - SQRT(...)

            if verbose:
                print(
                    f"DEBUG: Numerator options: {numerator_plus:.1f}, {numerator_minus:.1f}"
                )
                print(f"DEBUG: Denominator: {denominator:.1f}")

            # Calculate the arguments for ln() - this is the complete fraction inside ln()
            ln_arg_plus = numerator_plus / denominator
            ln_arg_minus = numerator_minus / denominator

            if verbose:
                print(f"DEBUG: ln() arguments: {ln_arg_plus:.6f}, {ln_arg_minus:.6f}")

            # Calculate B values using the natural logarithm
            B_plus = np.log(ln_arg_plus) if ln_arg_plus > 0 else np.nan
            B_minus = np.log(ln_arg_minus) if ln_arg_minus > 0 else np.nan

            if verbose:
                print(f"DEBUG: B options: {B_plus:.6f}, {B_minus:.6f}")

            # For decreasing ROS sensors, we want B > 0
            if not np.isnan(B_plus) and B_plus > 0:
                B = B_plus
                if verbose:
                    print(f"DEBUG: Using B_plus = {B:.6f}")
            elif not np.isnan(B_minus) and B_minus > 0:
                B = B_minus
                if verbose:
                    print(f"DEBUG: Using B_minus = {B:.6f}")
            else:
                print(f"ERROR: No valid positive B found")
                return {
                    "A": 0,
                    "B": 0,
                    "C": 0,
                }  # Calculate A using: A = (S1-S3)/(exp(-B*X1)-exp(-B*X3))
            exp_neg_BX1 = np.exp(-B * X1)
            exp_neg_BX3 = np.exp(-B * X3)

            print(f"DEBUG: Step 4 - Calculate A coefficient:")
            print(f"DEBUG:   exp(-B*X1) = exp(-{B:.6f}*{X1}) = {exp_neg_BX1:.6f}")
            print(f"DEBUG:   exp(-B*X3) = exp(-{B:.6f}*{X3}) = {exp_neg_BX3:.6f}")

            denominator_A = exp_neg_BX1 - exp_neg_BX3
            print(
                f"DEBUG:   Denominator for A = {exp_neg_BX1:.6f} - {exp_neg_BX3:.6f} = {denominator_A:.6f}"
            )

            if abs(denominator_A) < 1e-10:
                print(
                    f"ERROR: Denominator for A calculation too small: {denominator_A}"
                )
                return {"A": 0, "B": 0, "C": 0}

            A = (S1 - S3) / denominator_A
            print(
                f"DEBUG: A calculation: ({S1:.1f} - {S3:.1f}) / ({exp_neg_BX1:.6f} - {exp_neg_BX3:.6f}) = {A:.3f}"
            )

            # Calculate C using: C = S2 - A*exp(-B*X2)
            exp_neg_BX2 = np.exp(-B * X2)
            C = S2 - A * exp_neg_BX2
            print(
                f"DEBUG: C calculation: {S2:.1f} - {A:.3f} * {exp_neg_BX2:.6f} = {C:.3f}"
            )
            print(f"DEBUG: Final parameters - A={A:.3f}, B={B:.6f}, C={C:.3f}")

            # Verify the calculations using CORRECT model: S = A * exp(-B * X) + C
            S1_calc = A * np.exp(-B * X1) + C
            S2_calc = A * np.exp(-B * X2) + C
            S3_calc = A * np.exp(-B * X3) + C
            print(f"DEBUG: Verification using S = A * exp(-B * X) + C:")
            print(f"  S1: {S1:.1f} vs {S1_calc:.1f} (error: {abs(S1-S1_calc):.1f})")
            print(f"  S2: {S2:.1f} vs {S2_calc:.1f} (error: {abs(S2-S2_calc):.1f})")
            print(f"  S3: {S3:.1f} vs {S3_calc:.1f} (error: {abs(S3-S3_calc):.1f})")

            return {"A": A, "B": B, "C": C}

        except Exception as e:
            print(f"Error calculating calibration parameters: {e}")
            import traceback

            traceback.print_exc()
            return {"A": 0, "B": 0, "C": 0}

    def calculate_distance(self, sensor_reading, params):
        """Calculate distance from sensor reading using calibration parameters

        CORRECT Model: S = A * exp(-B * X) + C
        Solving for X (distance): X = -ln((S - C) / A) / B

        Where:
        - S = sensor_reading (raw sensor value)
        - A, B, C = calibration coefficients
        - X = distance (what we want to calculate)
        """
        try:
            A, B, C = params["A"], params["B"], params["C"]
            if A == 0 or B == 0:
                print(f"DEBUG: Invalid coefficients - A={A}, B={B}")
                return np.nan

            # Calculate the argument for the natural logarithm: (S - C) / A
            argument = (sensor_reading - C) / A
            if argument <= 0:
                print(
                    f"DEBUG: Invalid argument for ln: ({sensor_reading} - {C}) / {A} = {argument}"
                )
                return np.nan

            # Apply the correct formula: X = -ln((S - C) / A) / B
            distance = -np.log(argument) / B
            return distance
        except Exception as e:
            return np.nan

    def perform_calibrations(self):
        """Perform calibrations with position mispositions from -500μm to +500μm."""
        ideal_base_points = [1.0, 2.0, 3.0]  # mm targets

        # Generate shifts from -500μm to +500μm in 100μm steps
        shifts_mm = np.round(np.arange(-0.5, 0.5 + 0.1, 0.1), 3)

        # Ensure data is sorted by distance for interpolation
        sorted_data = self.data.sort_values("distance")
        xp = sorted_data["distance"].values
        fp = sorted_data["sensor_reading"].values

        print(f"\nDEBUG: Data range - Distance: {xp.min():.3f} to {xp.max():.3f} mm")
        print(f"DEBUG: Data range - Sensor: {fp.min():.0f} to {fp.max():.0f}")
        print(f"DEBUG: Data points: {len(xp)}")

        print(
            f"\n--- Performing Calibrations with Position Mispositions from -500μm to +500μm ---"
        )
        print(f"DEBUG: Shift values: {shifts_mm*1000} μm")

        # Initialize storage for raw calibration data
        self.raw_calibration_data = []

        for shift_mm in shifts_mm:
            target_distances = [p + shift_mm for p in ideal_base_points]

            print(
                f"\nDEBUG: Shift {shift_mm*1000:+.0f}μm - Target distances: {target_distances}"
            )

            # Check if target distances are within data range
            for i, target in enumerate(target_distances):
                if target < xp.min() or target > xp.max():
                    print(
                        f"WARNING: Target distance {target:.3f} mm is outside data range!"
                    )  # Use interpolation to get sensor readings for target distances
            sensor_readings = np.interp(target_distances, xp, fp)

            if shift_mm == 0.0:  # Only show detailed output for reference calibration
                print(f"DEBUG: Interpolated sensor readings: {sensor_readings}")
            # Calculate calibration parameters
            if shift_mm == 0.0:  # Only show detailed output for reference calibration
                params = self.calculate_calibration_parameters(
                    sensor_readings, verbose=True
                )
            else:
                params = self.calculate_calibration_parameters(
                    sensor_readings, verbose=False
                )

            # Store raw calibration data used for coefficient calculation
            raw_data_entry = {
                "Calibration": f"shift_{int(round(shift_mm * 1000)):+d}um",
                "Distance_1mm": target_distances[0],
                "Sensor_Reading_1mm": sensor_readings[0],
                "Distance_2mm": target_distances[1],
                "Sensor_Reading_2mm": sensor_readings[1],
                "Distance_3mm": target_distances[2],
                "Sensor_Reading_3mm": sensor_readings[2],
                "A_Coefficient": params["A"],
                "B_Coefficient": params["B"],
                "C_Coefficient": params["C"],
                "Notes": f"Raw sensor readings at exactly {target_distances[0]:.1f}, {target_distances[1]:.1f}, {target_distances[2]:.1f} mm used to calculate A, B, C coefficients",
            }
            self.raw_calibration_data.append(raw_data_entry)

            # Only show detailed test for reference calibration
            if shift_mm == 0.0:
                print("DEBUG: Testing calibration accuracy:")
                for i, (target_dist, sensor_reading) in enumerate(
                    zip(target_distances, sensor_readings)
                ):
                    predicted_dist = self.calculate_distance(sensor_reading, params)
                    error = predicted_dist - target_dist
                    print(
                        f"  Point {i+1}: Target={target_dist:.3f}, Sensor={sensor_reading:.0f}, Predicted={predicted_dist:.3f}, Error={error:.6f}"
                    )

            # Store calibration
            shift_um = int(round(shift_mm * 1000))
            calib_name = f"shift_{shift_um:+d}um"
            self.calibrations[calib_name] = {
                "params": params,
                "target_distances": target_distances,
                "sensor_readings": sensor_readings,
                "shift_mm": shift_mm,
            }

            print(
                f"Calibration '{calib_name}': A={params['A']:.2f}, B={params['B']:.6f}, C={params['C']:.2f}"
            )

    def calculate_errors(self):
        """Calculate distance errors for each calibration"""
        for calib_name, calib_data in self.calibrations.items():
            params = calib_data["params"]

            # Calculate predicted distances for all sensor readings
            predicted_distances = []
            for sensor_reading in self.data["sensor_reading"]:
                pred_dist = self.calculate_distance(sensor_reading, params)
                predicted_distances.append(pred_dist)

            predicted_distances = np.array(predicted_distances)
            actual_distances = self.data["distance"].values

            # Calculate errors
            errors = predicted_distances - actual_distances

            # Store original errors
            self.errors[calib_name] = errors

            # Calculate adjusted errors (remove mean bias)
            mean_error = np.nanmean(errors)
            adjusted_errors = errors - mean_error
            self.adjusted_errors[calib_name] = adjusted_errors

    def print_results(self):
        """Print calibration parameters and error statistics"""
        sensor_display_name = getattr(self, "sensor_name", self.sheet_name)
        print(f"\n=== SENSOR CALIBRATION ANALYSIS - {sensor_display_name} ===\n")
        print(f"Excel file: {self.excel_file}")
        print(f"Sheet: {self.sheet_name}")
        if hasattr(self, "unit_column"):
            print(f"Column: {self.unit_column}")
        print(f"Data increment detected: {self.increment} mm\n")

        print("Calibration Parameters:")
        print("-" * 70)
        print(f"{'Calibration':<15s} {'A':<15s} {'B':<12s} {'C':<15s}")
        print("-" * 70)
        for name, calib in self.calibrations.items():
            params = calib["params"]
            print(
                f"{name:<15s} {params['A']:<15.3f} {params['B']:<12.6f} {params['C']:<15.3f}"
            )

        # Modified Error Statistics section to show Non-Linearity (RMS of Adjusted Error)
        print(f"\nError Statistics (Non-Linearity Focus):")
        print("-" * 80)  # Adjusted width
        print(
            f"{'Calibration':<20s} {'Non-Lin (RMS Adj Err)':<22s} {'Max Orig Err':<18s} {'Mean Orig Err':<18s}"
        )
        print("-" * 80)  # Adjusted width

        for name, errors_arr in self.errors.items():
            adj_errors_arr = self.adjusted_errors[name]

            # Non-linearity is RMS of adjusted errors
            nonlinearity = np.sqrt(np.nanmean(adj_errors_arr**2))

            # Original error statistics
            max_orig_err = np.nanmax(np.abs(errors_arr))
            mean_orig_err = np.nanmean(errors_arr)

            print(
                f"{name:<20s} {nonlinearity:<22.6f} {max_orig_err:<18.6f} {mean_orig_err:<18.6f}"
            )

        print(f"\nAdjusted Error Statistics (Original Error - Mean Original Error):")
        print("-" * 70)
        print(
            f"{'Calibration':<15s} {'RMS Error':<12s} {'Max Error':<12s} {'Mean Error':<12s}"
        )
        print("-" * 70)
        for name, adj_errors_arr in self.adjusted_errors.items():
            rms_error = np.sqrt(np.nanmean(adj_errors_arr**2))
            max_error = np.nanmax(np.abs(adj_errors_arr))
            mean_error = np.nanmean(adj_errors_arr)

            print(
                f"{name:<15s} {rms_error:<12.6f} {max_error:<12.6f} {mean_error:<12.6f}"
            )

    def plot_errors(self):
        """Plot predicted distance vs true distance and prediction errors"""
        num_calibrations = len(self.errors)
        if num_calibrations == 0:
            print("No calibration data available for plotting.")
            return

        sensor_display_name = getattr(self, "sensor_name", self.sheet_name)

        fig, axs = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        # Use a colormap that provides distinct colors for many lines
        num_unique_calibs = len(self.calibrations)
        colors = plt.cm.nipy_spectral(np.linspace(0, 1, max(10, num_unique_calibs)))

        # Plot 1: Predicted Distance vs True Distance
        ax1 = axs[0]

        # Add perfect calibration line (y=x)
        ax1.plot(
            [0, 4],
            [0, 4],
            "k--",
            alpha=0.5,
            linewidth=2,
            label="Perfect Calibration (y=x)",
        )

        for i, (name, calib_data) in enumerate(self.calibrations.items()):
            color = colors[i % len(colors)]
            params = calib_data["params"]

            # Calculate predicted distances for all sensor readings
            predicted_distances = []
            for sensor_reading in self.data["sensor_reading"]:
                pred_dist = self.calculate_distance(sensor_reading, params)
                predicted_distances.append(pred_dist)

            ax1.plot(
                self.data["distance"],  # True distance (x-axis)
                predicted_distances,  # Predicted distance (y-axis)
                color=color,
                label=name,
                linewidth=1.5,
                alpha=0.8,
            )

        ax1.set_ylabel("Predicted Distance (mm)")
        ax1.set_title(f"Predicted vs True Distance - {sensor_display_name}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Prediction Error (Predicted - True) vs True Distance
        ax2 = axs[1]
        for i, (name, errors_arr) in enumerate(self.errors.items()):
            color = colors[i % len(colors)]
            ax2.plot(
                self.data["distance"],  # True distance (x-axis)
                errors_arr,  # Predicted - True distance (y-axis)
                color=color,
                label=name,
                linewidth=1.5,
                alpha=0.8,
            )

        # Add zero error reference line
        ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5, linewidth=1)

        ax2.set_xlabel("True Distance (mm)")
        ax2.set_ylabel("Prediction Error (mm)\n(Predicted Distance - True Distance)")
        ax2.set_title(f"Prediction Error vs True Distance - {sensor_display_name}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def save_results(self, output_dir=".", base_filename="calibration_results"):
        """Save results to Excel file"""
        try:
            output_path = os.path.join(output_dir, f"{base_filename}.xlsx")

            # Create summary data
            summary_data = []
            for name, calib in self.calibrations.items():
                params = calib["params"]
                errors_arr = self.errors[name]
                adj_errors_arr = self.adjusted_errors[name]

                summary_data.append(
                    {
                        "Calibration": name,
                        "A": params["A"],
                        "B": params["B"],
                        "C": params["C"],
                        "RMS_Adjusted_Error": np.sqrt(np.nanmean(adj_errors_arr**2)),
                        "Max_Original_Error": np.nanmax(np.abs(errors_arr)),
                        "Mean_Original_Error": np.nanmean(errors_arr),
                    }
                )

            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                # Write summary
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

                # Write raw calibration data used for coefficient calculation
                if hasattr(self, "raw_calibration_data") and self.raw_calibration_data:
                    raw_data_df = pd.DataFrame(self.raw_calibration_data)
                    raw_data_df.to_excel(
                        writer, sheet_name="Raw_Calibration_Data", index=False
                    )
                    print(
                        f"DEBUG: Added raw calibration data sheet with {len(self.raw_calibration_data)} entries"
                    )  # Write detailed errors
                error_data = pd.DataFrame({"Distance": self.data["distance"]})
                for name, errors_arr in self.errors.items():
                    error_data[f"{name}_Error"] = errors_arr
                    error_data[f"{name}_Adj_Error"] = self.adjusted_errors[name]

                error_data.to_excel(writer, sheet_name="Detailed_Errors", index=False)

                # Write original raw data from Excel file for reference
                if hasattr(self, "data") and self.data is not None:
                    original_data = self.data.copy()
                    # Add some metadata
                    if hasattr(self, "sensor_name"):
                        original_data["Sensor_Name"] = self.sensor_name
                    if hasattr(self, "excel_file"):
                        original_data["Source_File"] = os.path.basename(self.excel_file)
                    if hasattr(self, "sheet_name"):
                        original_data["Source_Sheet"] = self.sheet_name
                    if hasattr(self, "unit_column"):
                        original_data["Source_Column"] = self.unit_column

                    original_data.to_excel(
                        writer, sheet_name="Original_Raw_Data", index=False
                    )
                    print(
                        f"DEBUG: Added original raw data sheet with {len(original_data)} data points"
                    )

            print(f"Results saved to: {output_path}")

        except Exception as e:
            print(f"Error saving results: {e}")

    def plot_raw_and_interpolated_data(self):
        """Plot raw data and show interpolated points before analysis begins"""
        if self.data is None or self.data.empty:
            print("No data available for plotting.")
            return

        sensor_display_name = getattr(self, "sensor_name", self.sheet_name)

        # Store data for combined plotting - return data instead of plotting immediately
        return {
            "sensor_name": sensor_display_name,
            "distance": self.data["distance"],
            "sensor_reading": self.data["sensor_reading"],
            "data_stats": {
                "points": len(self.data),
                "distance_range": (
                    self.data["distance"].min(),
                    self.data["distance"].max(),
                ),
                "sensor_range": (
                    self.data["sensor_reading"].min(),
                    self.data["sensor_reading"].max(),
                ),
                "increment": self.increment,
            },
        }

    def plot_nonlinearity_vs_shift(self):
        """Plot non-linearity (RMS adjusted error) vs calibration misposition"""
        if not self.calibrations:
            print("No calibration data available for non-linearity plotting.")
            return

        sensor_display_name = getattr(self, "sensor_name", self.sheet_name)

        # Prepare data for plotting
        shifts_mm = []
        nonlinearity_all = []
        nonlinearity_1_3mm = []
        calib_names = []

        range_mask_1_3mm = (self.data["distance"] >= 1.0) & (
            self.data["distance"] <= 3.0
        )

        for name, calib_info in self.calibrations.items():
            # Extract shift from calibration name (assumes format "shift_XXXum")
            try:
                shift_str = name.split("_")[1].replace("um", "")
                shift_um = float(shift_str)
                shift_mm = shift_um / 1000.0
                shifts_mm.append(shift_um)  # Keep in micrometers for x-axis
                calib_names.append(name)

                # Calculate non-linearity (RMS of adjusted errors)
                adj_errors = self.adjusted_errors[name]
                nonlinearity_all.append(np.sqrt(np.mean(adj_errors**2)))

                # Calculate non-linearity for 1-3mm range
                adj_errors_1_3mm = adj_errors[range_mask_1_3mm]
                if len(adj_errors_1_3mm) > 0:
                    nonlinearity_1_3mm.append(np.sqrt(np.mean(adj_errors_1_3mm**2)))
                else:
                    nonlinearity_1_3mm.append(0)

            except (IndexError, ValueError):
                continue

        if not shifts_mm:
            print("Could not extract shift information from calibration names.")
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
            "o-",  # fmt string specifies marker
            markersize=8,
            linewidth=2,
            alpha=0.8,
            label="All Data Range",
        )
        ax.plot(
            shifts_mm,
            nonlinearity_1_3mm,
            "s-",  # fmt string specifies marker
            markersize=8,
            linewidth=2,
            alpha=0.8,
            label="1-3mm Range",
        )

        ax.set_xlabel("Calibration Misposition (μm)", fontsize=12)
        ax.set_ylabel("Non-Linearity (RMS Adjusted Error, mm)", fontsize=12)
        ax.set_title(
            f"Non-Linearity vs Calibration Misposition - {sensor_display_name}",
            fontsize=14,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)

        # Add some formatting
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_nonlinearity_comparison_with_offset_reduction(self):
        """Plot comparison showing how offset reduction affects non-linearity patterns"""
        if not self.calibrations:
            print("No calibration data available for comparison plotting.")
            return

        sensor_display_name = getattr(self, "sensor_name", self.sheet_name)

        fig, ax = plt.subplots(1, 1, figsize=(16, 10))

        # Define distinct colors and markers for better visibility
        distinct_colors = [
            "#e6194B",
            "#3cb44b",
            "#4363d8",
            "#f58231",
            "#911eb4",
            "#42d4f4",
            "#f032e6",
            "#ffe119",
            "#bfef45",
            "#fabed4",
            "#000075",
        ]
        markers = ["o", "s", "^", "D", "v", "*", "p", "h", "X", "+"]

        # Get sorted calibration names by shift value
        calib_names = sorted(
            self.calibrations.keys(),
            key=lambda x: (
                float(x.split("_")[1].replace("um", "")) if "shift_" in x else 0
            ),
        )

        # Plot each calibration's offset-reduced errors
        for idx, calib_name in enumerate(calib_names):
            if calib_name not in self.errors:
                continue

            errors = self.errors[calib_name]
            distances = self.data["distance"].values
            mean_error = np.mean(errors)
            offset_reduced_errors = errors - mean_error

            # Extract shift value for the label
            shift_value = int(calib_name.split("_")[1].replace("um", ""))

            ax.plot(
                distances,
                offset_reduced_errors,
                linewidth=2,
                marker=markers[idx % len(markers)],
                markersize=4,
                alpha=0.8,
                color=distinct_colors[idx % len(distinct_colors)],
                label=f"{shift_value:+d}μm (Offset: {mean_error:.3f}mm)",
            )

        # Add reference line at y=0
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)

        ax.set_xlabel("Distance (mm)", fontsize=12)
        ax.set_ylabel("Offset-Reduced Error (mm)", fontsize=12)
        ax.set_title(
            f"Non-Linearity Comparison After Offset Reduction - {sensor_display_name}",
            fontsize=14,
        )
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(title="Misposition", fontsize=10, title_fontsize=11)

        plt.tight_layout()
        plt.show()

    def plot_nonlinearity_heatmap(all_sensor_analyzers):
        """Create a heatmap showing non-linearity scores for all sensors vs calibration shifts"""
        if not all_sensor_analyzers:
            print("No sensor data available for heatmap.")
            return

        # Collect data for heatmap
        sensor_names = []
        all_shifts = set()

        # First pass: collect all unique shifts and sensor names
        for analyzer in all_sensor_analyzers:
            sensor_name = getattr(analyzer, "sensor_name", analyzer.sheet_name)
            sensor_names.append(sensor_name)

            for name in analyzer.adjusted_errors.keys():
                try:
                    shift_str = name.split("_")[1].replace("um", "")
                    shift_um = float(shift_str)
                    all_shifts.add(shift_um)
                except (IndexError, ValueError):
                    continue

        # Sort shifts for consistent ordering
        sorted_shifts = sorted(all_shifts)

        # Create matrix for heatmap data
        heatmap_data = np.full((len(sensor_names), len(sorted_shifts)), np.nan)

        # Second pass: fill the matrix with non-linearity scores
        for sensor_idx, analyzer in enumerate(all_sensor_analyzers):
            for name, adj_errors in analyzer.adjusted_errors.items():
                try:
                    shift_str = name.split("_")[1].replace("um", "")
                    shift_um = float(shift_str)
                    shift_idx = sorted_shifts.index(shift_um)

                    # Calculate non-linearity (RMS of adjusted errors)
                    nonlinearity = np.sqrt(np.nanmean(adj_errors**2))
                    heatmap_data[sensor_idx, shift_idx] = nonlinearity

                except (IndexError, ValueError):
                    continue

        # Create the heatmap
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Create heatmap with intuitive colormap (green=low/good, red=high/bad)
        im = ax.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
        
        # Set ticks and labels
        ax.set_xticks(range(len(sorted_shifts)))
        ax.set_xticklabels([f"{int(shift):+d}" for shift in sorted_shifts], rotation=45)
        ax.set_yticks(range(len(sensor_names)))
        ax.set_yticklabels(sensor_names)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Non-Linearity (RMS Adjusted Error, mm)', fontsize=12)
        
        # Add value annotations on heatmap with adaptive text color
        for i in range(len(sensor_names)):
            for j in range(len(sorted_shifts)):
                if not np.isnan(heatmap_data[i, j]):
                    # Use black text for better readability on the RdYlGn_r colormap
                    text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=9, fontweight='bold')

        # Labels and title
        ax.set_xlabel('Calibration Misposition (μm)', fontsize=12)
        ax.set_ylabel('ROS Sensors', fontsize=12)
        ax.set_title('Non-Linearity Heatmap - All ROS Sensors vs Calibration Misposition', fontsize=14)
        
        # Add grid
        ax.set_xticks(np.arange(len(sorted_shifts))-0.5, minor=True)
        ax.set_yticks(np.arange(len(sensor_names))-0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", size=0)

        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"\n=== NON-LINEARITY HEATMAP SUMMARY ===")
        print(f"Sensors analyzed: {len(sensor_names)}")
        print(f"Calibration shifts: {len(sorted_shifts)} (from {int(min(sorted_shifts))}μm to {int(max(sorted_shifts))}μm)")
        
        # Find best and worst performing sensors
        sensor_avg_nonlinearity = np.nanmean(heatmap_data, axis=1)
        best_sensor_idx = np.nanargmin(sensor_avg_nonlinearity)
        worst_sensor_idx = np.nanargmax(sensor_avg_nonlinearity)
        
        print(f"\nSensor Performance Summary:")
        print(f"Best performing sensor: {sensor_names[best_sensor_idx]} (avg non-linearity: {sensor_avg_nonlinearity[best_sensor_idx]:.4f}mm)")
        print(f"Worst performing sensor: {sensor_names[worst_sensor_idx]} (avg non-linearity: {sensor_avg_nonlinearity[worst_sensor_idx]:.4f}mm)")
        
        # Find best and worst calibration positions
        position_avg_nonlinearity = np.nanmean(heatmap_data, axis=0)
        best_position_idx = np.nanargmin(position_avg_nonlinearity)
        worst_position_idx = np.nanargmax(position_avg_nonlinearity)
        
        print(f"\nCalibration Position Performance Summary:")
        print(f"Best position: {int(sorted_shifts[best_position_idx]):+d}μm (avg non-linearity: {position_avg_nonlinearity[best_position_idx]:.4f}mm)")
        print(f"Worst position: {int(sorted_shifts[worst_position_idx]):+d}μm (avg non-linearity: {position_avg_nonlinearity[worst_position_idx]:.4f}mm)")


# Add these functions right after the class definition and before the if __name__ == "__main__": section

def plot_all_sensors_raw_data(sensor_data_list):
    """Plot raw data for all sensors on one combined graph"""
    if not sensor_data_list:
        print("No sensor data available for plotting.")
        return

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # Define colors for different sensors
    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]

    for i, sensor_data in enumerate(sensor_data_list):
        color = colors[i % len(colors)]
        sensor_name = sensor_data["sensor_name"]

        # Plot raw data as scatter
        ax.scatter(
            sensor_data["distance"],
            sensor_data["sensor_reading"],
            c=color,
            s=20,
            alpha=0.6,
            label=f"{sensor_name} - Raw Data",
            zorder=3,
        )

        # Create interpolation curve for visualization
        distance_range = np.linspace(
            sensor_data["distance"].min(), sensor_data["distance"].max(), 500
        )

        # Ensure data is sorted for interpolation
        sorted_indices = np.argsort(sensor_data["distance"])
        xp = sensor_data["distance"].iloc[sorted_indices].values
        fp = sensor_data["sensor_reading"].iloc[sorted_indices].values

        interpolated_curve = np.interp(distance_range, xp, fp)
        ax.plot(
            distance_range,
            interpolated_curve,
            color=color,
            linewidth=2,
            alpha=0.8,
            linestyle="-",
            label=f"{sensor_name} - Interpolation",
            zorder=2,
        )

    ax.set_xlabel("True Distance (mm)", fontsize=12)
    ax.set_ylabel("Sensor Reading", fontsize=12)
    ax.set_title("Raw Data and Interpolation Overview - All ROS Sensors", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    plt.show()

def plot_predicted_vs_true_all_sensors(all_sensor_analyzers):
    """Plot predicted vs true distance - 4 subplots, one per sensor"""
    if not all_sensor_analyzers:
        print("No sensor data available for plotting.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # Use colormap for different calibrations
    num_calibrations = len(next(iter(all_sensor_analyzers[0].calibrations.keys() if all_sensor_analyzers else []), []))
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, max(10, num_calibrations)))

    for sensor_idx, analyzer in enumerate(all_sensor_analyzers):
        if sensor_idx >= 4:  # Only handle 4 sensors
            break
            
        ax = axes[sensor_idx]
        sensor_name = getattr(analyzer, "sensor_name", analyzer.sheet_name)
        
        # Add perfect calibration line
        ax.plot([0, 4], [0, 4], "k--", alpha=0.5, linewidth=2, label="Perfect Calibration (y=x)")
        
        # Plot all calibrations for this sensor
        for i, (name, calib_data) in enumerate(analyzer.calibrations.items()):
            color = colors[i % len(colors)]
            params = calib_data["params"]

            # Calculate predicted distances for all sensor readings
            predicted_distances = []
            for sensor_reading in analyzer.data["sensor_reading"]:
                pred_dist = analyzer.calculate_distance(sensor_reading, params)
                predicted_distances.append(pred_dist)

            ax.plot(
                analyzer.data["distance"],  # True distance (x-axis)
                predicted_distances,  # Predicted distance (y-axis)
                color=color,
                label=name,
                linewidth=1.5,
                alpha=0.8,
            )

        ax.set_xlabel("True Distance (mm)")
        ax.set_ylabel("Predicted Distance (mm)")
        ax.set_title(f"Predicted vs True Distance - {sensor_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(all_sensor_analyzers), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_prediction_errors_all_sensors(all_sensor_analyzers):
    """Plot prediction errors - 4 subplots, one per sensor"""
    if not all_sensor_analyzers:
        print("No sensor data available for plotting.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # Use colormap for different calibrations
    num_calibrations = len(next(iter(all_sensor_analyzers[0].calibrations.keys() if all_sensor_analyzers else []), []))
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, max(10, num_calibrations)))

    for sensor_idx, analyzer in enumerate(all_sensor_analyzers):
        if sensor_idx >= 4:  # Only handle 4 sensors
            break
            
        ax = axes[sensor_idx]
        sensor_name = getattr(analyzer, "sensor_name", analyzer.sheet_name)
        
        # Plot all prediction errors for this sensor
        for i, (name, errors_arr) in enumerate(analyzer.errors.items()):
            color = colors[i % len(colors)]
            ax.plot(
                analyzer.data["distance"],  # True distance (x-axis)
                errors_arr,  # Predicted - True distance (y-axis)
                color=color,
                label=name,
                linewidth=1.5,
                alpha=0.8,
            )

        # Add zero error reference line
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.5, linewidth=1)
        
        ax.set_xlabel("True Distance (mm)")
        ax.set_ylabel("Prediction Error (mm)")
        ax.set_title(f"Prediction Error vs True Distance - {sensor_name}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(all_sensor_analyzers), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_nonlinearity_vs_shift_all_sensors(all_sensor_analyzers):
    """Plot non-linearity vs calibration shift - 4 subplots, one per sensor"""
    if not all_sensor_analyzers:
        print("No sensor data available for plotting.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    for sensor_idx, analyzer in enumerate(all_sensor_analyzers):
        if sensor_idx >= 4:  # Only handle 4 sensors
            break
            
        ax = axes[sensor_idx]
        sensor_name = getattr(analyzer, "sensor_name", analyzer.sheet_name)
        
        # Calculate non-linearity for each shift
        shifts_um = []
        nonlinearity_all = []
        nonlinearity_1_3mm = []
        
        range_mask_1_3mm = (analyzer.data["distance"] >= 1.0) & (analyzer.data["distance"] <= 3.0)

        for name, adj_errors in analyzer.adjusted_errors.items():
            try:
                shift_str = name.split("_")[1].replace("um", "")
                shift_um = float(shift_str)
                shifts_um.append(shift_um)
                
                # All data range
                nonlinearity = np.sqrt(np.nanmean(adj_errors**2))
                nonlinearity_all.append(nonlinearity)
                
                # 1-3mm range
                adj_errors_1_3mm = adj_errors[range_mask_1_3mm]
                if len(adj_errors_1_3mm) > 0:
                    nonlinearity_1_3mm.append(np.sqrt(np.nanmean(adj_errors_1_3mm**2)))
                else:
                    nonlinearity_1_3mm.append(0)
                    
            except (IndexError, ValueError):
                continue

        if shifts_um:
            # Sort by shift for proper line plotting
            sorted_indices = np.argsort(shifts_um)
            shifts_sorted = [shifts_um[idx] for idx in sorted_indices]
            nonlinearity_all_sorted = [nonlinearity_all[idx] for idx in sorted_indices]
            nonlinearity_1_3mm_sorted = [nonlinearity_1_3mm[idx] for idx in sorted_indices]
            
            # Plot both datasets
            ax.plot(
                shifts_sorted,
                nonlinearity_all_sorted,
                "o-",
                markersize=6,
                linewidth=2,
                alpha=0.8,
                label="All Data Range",
            )
            ax.plot(
                shifts_sorted,
                nonlinearity_1_3mm_sorted,
                "s-",
                markersize=6,
                linewidth=2,
                alpha=0.8,
                label="1-3mm Range",
            )

        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("Calibration Misposition (μm)")
        ax.set_ylabel("Non-Linearity (RMS Adjusted Error, mm)")
        ax.set_title(f"Non-Linearity vs Calibration Misposition - {sensor_name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for i in range(len(all_sensor_analyzers), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_offset_reduced_errors_all_sensors(all_sensor_analyzers):
    """Plot offset-reduced errors - 4 subplots, one per sensor"""
    if not all_sensor_analyzers:
        print("No sensor data available for plotting.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # Define distinct colors and markers
    distinct_colors = [
        "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", 
        "#42d4f4", "#f032e6", "#ffe119", "#bfef45", "#fabed4", "#000075"
    ]
    markers = ["o", "s", "^", "D", "v", "*", "p", "h", "X", "+"]

    for sensor_idx, analyzer in enumerate(all_sensor_analyzers):
        if sensor_idx >= 4:  # Only handle 4 sensors
            break
            
        ax = axes[sensor_idx]
        sensor_name = getattr(analyzer, "sensor_name", analyzer.sheet_name)
        
        # Get sorted calibration names
        calib_names = sorted(
            analyzer.calibrations.keys(),
            key=lambda x: (float(x.split("_")[1].replace("um", "")) if "shift_" in x else 0)
        )
        
        # Plot each calibration's offset-reduced errors
        for idx, calib_name in enumerate(calib_names):
            if calib_name not in analyzer.errors:
                continue

            errors = analyzer.errors[calib_name]
            distances = analyzer.data["distance"].values
            mean_error = np.mean(errors)
            offset_reduced_errors = errors - mean_error

            # Extract shift value
            shift_value = int(calib_name.split("_")[1].replace("um", ""))

            ax.plot(
                distances,
                offset_reduced_errors,
                linewidth=2,
                marker=markers[idx % len(markers)],
                markersize=4,
                alpha=0.8,
                color=distinct_colors[idx % len(distinct_colors)],
                label=f"{shift_value:+d}μm (Offset: {mean_error:.3f}mm)",
            )

        # Add reference line at y=0
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        
        ax.set_xlabel("Distance (mm)")
        ax.set_ylabel("Offset-Reduced Error (mm)")
        ax.set_title(f"Non-Linearity After Offset Reduction - {sensor_name}")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(title="Misposition", fontsize=8, title_fontsize=9)
    
    # Hide unused subplots
    for i in range(len(all_sensor_analyzers), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_nonlinearity_heatmap(all_sensor_analyzers):
    """Create a heatmap showing non-linearity scores for all sensors vs calibration shifts"""
    if not all_sensor_analyzers:
        print("No sensor data available for heatmap.")
        return

    # Collect data for heatmap
    sensor_names = []
    all_shifts = set()
    
    # First pass: collect all unique shifts and sensor names
    for analyzer in all_sensor_analyzers:
        sensor_name = getattr(analyzer, "sensor_name", analyzer.sheet_name)
        sensor_names.append(sensor_name)
        
        for name in analyzer.adjusted_errors.keys():
            try:
                shift_str = name.split("_")[1].replace("um", "")
                shift_um = float(shift_str)
                all_shifts.add(shift_um)
            except (IndexError, ValueError):
                continue
    
    # Sort shifts for consistent ordering
    sorted_shifts = sorted(all_shifts)
    
    # Create matrix for heatmap data
    heatmap_data = np.full((len(sensor_names), len(sorted_shifts)), np.nan)
    
    # Second pass: fill the matrix with non-linearity scores
    for sensor_idx, analyzer in enumerate(all_sensor_analyzers):
        for name, adj_errors in analyzer.adjusted_errors.items():
            try:
                shift_str = name.split("_")[1].replace("um", "")
                shift_um = float(shift_str)
                shift_idx = sorted_shifts.index(shift_um)
                
                # Calculate non-linearity (RMS of adjusted errors)
                nonlinearity = np.sqrt(np.nanmean(adj_errors**2))
                heatmap_data[sensor_idx, shift_idx] = nonlinearity

            except (IndexError, ValueError):
                continue
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create heatmap with intuitive colormap (green=low/good, red=high/bad)
    im = ax.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(len(sorted_shifts)))
    ax.set_xticklabels([f"{int(shift):+d}" for shift in sorted_shifts], rotation=45)
    ax.set_yticks(range(len(sensor_names)))
    ax.set_yticklabels(sensor_names)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Non-Linearity (RMS Adjusted Error, mm)', fontsize=12)
    
    # Add value annotations on heatmap with adaptive text color
    for i in range(len(sensor_names)):
        for j in range(len(sorted_shifts)):
            if not np.isnan(heatmap_data[i, j]):
                # Use black text for better readability on the RdYlGn_r colormap
                text = ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=9, fontweight='bold')

    # Labels and title
    ax.set_xlabel('Calibration Misposition (μm)', fontsize=12)
    ax.set_ylabel('ROS Sensors', fontsize=12)
    ax.set_title('Non-Linearity Heatmap - All ROS Sensors vs Calibration Misposition', fontsize=14)
    
    # Add grid
    ax.set_xticks(np.arange(len(sorted_shifts))-0.5, minor=True)
    ax.set_yticks(np.arange(len(sensor_names))-0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)

    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n=== NON-LINEARITY HEATMAP SUMMARY ===")
    print(f"Sensors analyzed: {len(sensor_names)}")
    print(f"Calibration shifts: {len(sorted_shifts)} (from {int(min(sorted_shifts))}μm to {int(max(sorted_shifts))}μm)")
    
    # Find best and worst performing sensors
    sensor_avg_nonlinearity = np.nanmean(heatmap_data, axis=1)
    best_sensor_idx = np.nanargmin(sensor_avg_nonlinearity)
    worst_sensor_idx = np.nanargmax(sensor_avg_nonlinearity)
    
    print(f"\nSensor Performance Summary:")
    print(f"Best performing sensor: {sensor_names[best_sensor_idx]} (avg non-linearity: {sensor_avg_nonlinearity[best_sensor_idx]:.4f}mm)")
    print(f"Worst performing sensor: {sensor_names[worst_sensor_idx]} (avg non-linearity: {sensor_avg_nonlinearity[worst_sensor_idx]:.4f}mm)")
    
    # Find best and worst calibration positions
    position_avg_nonlinearity = np.nanmean(heatmap_data, axis=0)
    best_position_idx = np.nanargmin(position_avg_nonlinearity)
    worst_position_idx = np.nanargmax(position_avg_nonlinearity)
    
    print(f"\nCalibration Position Performance Summary:")
    print(f"Best position: {int(sorted_shifts[best_position_idx]):+d}μm (avg non-linearity: {position_avg_nonlinearity[best_position_idx]:.4f}mm)")
    print(f"Worst position: {int(sorted_shifts[worst_position_idx]):+d}μm (avg non-linearity: {position_avg_nonlinearity[worst_position_idx]:.4f}mm)")


# Replace the existing main execution section with this updated version:
if __name__ == "__main__":
    # --- CONFIGURATION ---
    excel_file_path = r"C:\Users\geller\OneDrive - HP Inc\data\ROS\using ROS for cast iron\ROS vs PIP tsrget in Tamar104 press\summary.xlsx"

    # First, read the header row to get actual sensor names from 'all raw' sheet
    try:
        df_header = pd.read_excel(excel_file_path, sheet_name="all raw")
        print("Raw header data:")
        print(df_header.head(3))  # Show first 3 rows to debug

        print("Header information from row 2 of 'all raw' sheet:")
        if len(df_header) > 1:
            sensor_names = {}
            # Read sensor names from columns B, C, D, E (indices 1, 2, 3, 4)
            for col_letter, col_index in [("B", 1), ("C", 2), ("D", 3), ("E", 4)]:
                if col_index < df_header.shape[1]:
                    sensor_name = df_header.iloc[1, col_index]  # Row 2 (index 1)

                    # Check if we got a valid sensor name (not a number)
                    if (
                        pd.isna(sensor_name)
                        or str(sensor_name).replace(".", "").replace(",", "").isdigit()
                    ):
                        # Try row 1 if row 2 has numbers
                        if len(df_header) > 0:
                            sensor_name = df_header.iloc[0, col_index]

                        # If still not good, use default
                        if (
                            pd.isna(sensor_name)
                            or str(sensor_name)
                            .replace(".", "")
                            .replace(",", "")
                            .isdigit()
                        ):
                            sensor_name = f"Sensor_{col_letter}"

                    sensor_names[col_letter] = str(sensor_name).strip()
                    print(f"  Column {col_letter}: {sensor_names[col_letter]}")
                else:
                    sensor_names[col_letter] = f"Sensor_{col_letter}"
                    print(
                        f"  Column {col_letter}: {sensor_names[col_letter]} (default)"
                    )
        else:
            print("Could not read header row, using default names")
            sensor_names = {
                "B": "Sensor_B",
                "C": "Sensor_C",
                "D": "Sensor_D",
                "E": "Sensor_E",
            }
    except Exception as e:
        print(f"Error reading header: {e}")
        sensor_names = {
            "B": "Sensor_B",
            "C": "Sensor_C",
            "D": "Sensor_D",
            "E": "Sensor_E",
        }  # Define only column B for debugging purposes
    ros_channels = [
        {
            "name": sensor_names["B"],  # Uses actual sensor name from Excel
            "sheet": "all raw",
            "column": "B",
        },
        # Uncommented other channels for full analysis
        {
            "name": sensor_names["C"],  # Uses actual sensor name from Excel
            "sheet": "all raw",
            "column": "C",
        },
        {
            "name": sensor_names["D"],  # Uses actual sensor name from Excel
            "sheet": "all raw",
            "column": "D",
        },
        {
            "name": sensor_names["E"],  # Uses actual sensor name from Excel
            "sheet": "all raw",
            "column": "E",
        },
    ]

    output_directory = r"C:\Users\geller\OneDrive - HP Inc\data\ROS\using ROS for cast iron\ROS vs PIP tsrget in Tamar104 press"
    base_output_filename_prefix = "tamar104_analysis"

    # --- CREATE OUTPUT DIRECTORY IF IT DOESN'T EXIST ---
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")

    # Verify the data structure first
    print(f"\n=== VERIFYING DATA STRUCTURE ===")
    print(f"File: {excel_file_path}")
    print(f"Sheet: all raw")
    print(f"Expected structure:")
    print(f"  Column A: Distance values")
    print(f"  Columns B-E: Raw sensor data for 4 sensors")
    print(f"  Row 1: Headers")
    print(f"  Row 2: Sensor names")
    print(f"  Row 3+: Actual data")

    # Test data loading with a sample analyzer
    test_analyzer = SensorCalibrationAnalyzer(
        excel_file_path, "all raw", unit_column="B"
    )

    if test_analyzer.data is not None and not test_analyzer.data.empty:
        print(f"✓ Data loaded successfully!")
        print(f"  Total data points: {len(test_analyzer.data)}")
        print(
            f"  Distance range: {test_analyzer.data['distance'].min():.3f} - {test_analyzer.data['distance'].max():.3f} mm"
        )
        print(
            f"  Sensor reading range: {test_analyzer.data['sensor_reading'].min():.0f} - {test_analyzer.data['sensor_reading'].max():.0f}"
        )

        # Check data filtering
        filtered_data = test_analyzer.data[test_analyzer.data["distance"] <= 4]
        print(f"  Data points with distance ≤ 4mm: {len(filtered_data)}")

        if len(filtered_data) > 0:
            print(
                f"  Filtered distance range: {filtered_data['distance'].min():.3f} - {filtered_data['distance'].max():.3f} mm"
            )
        else:
            print("  ⚠️  No data points found with distance ≤ 4mm")
    else:
        print("❌ Failed to load data. Check file path and sheet name.")
        print("Available sheets in the file:")
        try:
            xl_file = pd.ExcelFile(excel_file_path)
            for sheet in xl_file.sheet_names:
                print(f"  - {sheet}")
        except Exception as e:
            print(f"  Error reading file: {e}")
        exit(1)  # --- EXECUTION FOR EACH ROS CHANNEL ---
    print(f"\n=== PROCESSING ALL ROS CHANNELS ===")

    # Store sensor data for combined plotting
    all_sensor_data = []

    # Store all analyzer instances for comprehensive plotting
    all_analyzers = []

    for i, ros_channel in enumerate(ros_channels, 1):
        channel_name = ros_channel["name"]
        sheet_name = ros_channel["sheet"]
        column = ros_channel["column"]

        print(
            f"\n--- Sensor {i}/{len(ros_channels)}: {channel_name} (Column {column}) ---"
        )

        analyzer = SensorCalibrationAnalyzer(
            excel_file_path, sheet_name, unit_column=column
        )

        if analyzer.data is not None and not analyzer.data.empty:
            print(f"Data points loaded: {len(analyzer.data)}")

            if len(analyzer.data) == 0:
                print(f"No data available for {channel_name}. Skipping.")
                continue

            # Store sensor data for combined plotting
            sensor_plot_data = analyzer.plot_raw_and_interpolated_data()

            all_sensor_data.append(sensor_plot_data)

            # Store analyzer instance for later comprehensive plotting
            all_analyzers.append(analyzer)

            # --- CALIBRATION AND ERROR ANALYSIS ---
            print(f"\n--- Calibration and Error Analysis for {channel_name} ---")
            analyzer.perform_calibrations()
            analyzer.calculate_errors()

            # Print calibration results
            analyzer.print_results()

            # Save results to Excel
            output_filename = f"{base_output_filename_prefix}_{channel_name}.xlsx"
            analyzer.save_results(output_dir=output_directory, base_filename=output_filename)

            print(f"Results saved to: {output_filename}")

            # --- INDIVIDUAL PLOTTING (COMMENTED OUT) ---
            # # Plot raw data and interpolation
            # plot_data = analyzer.plot_raw_and_interpolated_data()
            # fig, ax = plt.subplots(figsize=(14, 8))
            # ax.scatter(
            #     plot_data["distance"],
            #     plot_data["sensor_reading"],
            #     c="blue",
            #     s=20,
            #     alpha=0.6,
            #     label="Raw Data",
            # )
            # ax.plot(
            #     plot_data["distance"],
            #     np.interp(plot_data["distance"], plot_data["distance"], plot_data["sensor_reading"]),
            #     color="blue",
            #     linewidth=2,
            #     alpha=0.8,
            #     label="Interpolation",
            # )
            # ax.set_xlabel("True Distance (mm)", fontsize=12)
            # ax.set_ylabel("Sensor Reading", fontsize=12)
            # ax.set_title(f"Raw Data and Interpolation - {channel_name}", fontsize=14)
            # ax.legend(fontsize=10)
            # ax.grid(True, alpha=0.3)
            # plt.tight_layout()
            # plt.show()

            # # Plot predicted vs true distance
            # analyzer.plot_errors()

            # # Plot non-linearity vs shift
            # analyzer.plot_nonlinearity_vs_shift()

            # # Plot non-linearity comparison with offset reduction
            # analyzer.plot_nonlinearity_comparison_with_offset_reduction()

            # --- INDIVIDUAL SENSOR ANALYSIS COMPLETE ---
        else:
            print(f"❌ Failed to load data for {channel_name}. Skipping.")

    # --- COMBINED ANALYSIS FOR ALL SENSORS ---
    print(f"\n=== COMBINED ANALYSIS FOR ALL SENSORS ===")

    # Plot all sensors raw data
    plot_all_sensors_raw_data(all_sensor_data)

    # Plot predicted vs true distance for all sensors
    plot_predicted_vs_true_all_sensors(all_analyzers)

    # Plot prediction errors for all sensors
    plot_prediction_errors_all_sensors(all_analyzers)

    # Plot non-linearity vs shift for all sensors
    plot_nonlinearity_vs_shift_all_sensors(all_analyzers)

    # Plot offset-reduced errors for all sensors
    plot_offset_reduced_errors_all_sensors(all_analyzers)

    # Plot non-linearity heatmap for all sensors
    plot_nonlinearity_heatmap(all_analyzers)

    print(f"\n=== ANALYSIS COMPLETE ===")