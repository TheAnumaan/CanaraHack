import os
import pandas as pd
import dask.dataframe as dd
import io

# Root directory
root_dir = "/Users/kkartikaggarwal/Desktop/coding folder2/canrabank/HuMI" # <-- IMPORTANT: Adjust this path if needed
output_dir = "/Users/kkartikaggarwal/Desktop/coding folder2/canrabank/new" # Directory to store user-specific CSVs (changed name)
os.makedirs(output_dir, exist_ok=True) # Create the output directory if it doesn't exist

# Known column mappings for specific CSVs
# These are the *desired* column names. The script will try to apply them.
column_mappings = {
    "bluetooth.csv": ["timestamp(ms)", "bt_name", "bt_mac"],
    "gps.csv": ["timestamp(ms)", "orientation", "latitude", "longitude", "altitude", "bearing", "accuracy"],
    "wifi.csv": ["timestamp(ms)", "wifi_name", "wifi_level", "wifi_info", "wifi_channel", "wifi_freq"],
    "sensor_gyro.csv": ["timestamp(ms)", "orientation", "gyro_x", "gyro_y", "gyro_z"],
    "sensor_grav.csv": ["timestamp(ms)", "orientation", "gravity_data"],
    "sensor_humd.csv": ["timestamp(ms)", "orientation", "humidity"],
    "sensor_lacc.csv": ["timestamp(ms)", "orientation", "lacc_x", "lacc_y", "lacc_z"],
    "sensor_ligh.csv": ["timestamp(ms)", "orientation", "light_level"],
    "sensor_magn.csv": ["timestamp(ms)", "orientation", "magn_x", "magn_y", "magn_z"],
    "sensor_nacc.csv": ["timestamp(ms)", "orientation", "nacc_x", "nacc_y", "nacc_z"],
    "sensor_prox.csv": ["timestamp(ms)", "orientation", "proximity"],
    "sensor_temp.csv": ["timestamp(ms)", "orientation", "temperature"],
    "key_data.csv": ["timestamp(ms)", "key_field", "ascii_code"],
    "swipe.csv": ["timestamp(ms)", "orientation", "x", "y", "pressure", "action"],
    "scroll_X_touch.csv": ["timestamp(ms)", "orientation", "x", "y", "pressure", "action"],
    "touch_touch.csv": ["timestamp(ms)", "orientation", "x", "y", "pressure", "action"],
    "f_X_touch.csv": ["timestamp(ms)", "orientation", "x", "y", "pressure", "action"],
}

# Define potential delimiters to try, order matters (most common first)
DELIMITERS = [',', ';', '\t']

def infer_delimiter(file_path):
    """Infers the delimiter by trying to read a few lines with different delimiters."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_lines = [f.readline() for _ in range(5)]

        for delim in DELIMITERS:
            consistent_count = True
            first_line_cols = -1
            for line in first_lines:
                if line.strip():
                    cols = line.strip().split(delim)
                    if first_line_cols == -1:
                        first_line_cols = len(cols)
                    elif len(cols) != first_line_cols and len(cols) > 1:
                        consistent_count = False
                        break
            if consistent_count and first_line_cols > 1:
                return delim
    except Exception:
        pass
    return ','

def process_csv_to_dask_df(file_path, user, session, task, relative_name):
    base_file = os.path.basename(file_path)
    expected_cols = column_mappings.get(base_file)

    original_rows_count = 0
    try:
        # Get original line count (rough estimate, handles empty lines too)
        with open(file_path, 'r', encoding='utf-8') as f:
            original_rows_count = sum(1 for line in f if line.strip()) # Count non-empty lines
        if original_rows_count > 0:
            print(f"DEBUG: Processing {file_path} (Original non-empty lines: {original_rows_count})")
        else:
            print(f"DEBUG: Skipping {file_path} - file is empty or contains only whitespace.")
            return None

        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"DEBUG: Skipping {file_path} - file does not exist or is effectively empty.")
            return None

        inferred_delimiter = infer_delimiter(file_path)
        
        df_pandas = None
        try:
            df_pandas = pd.read_csv(file_path, sep=inferred_delimiter, engine='python', on_bad_lines='skip')
            if df_pandas.empty or (len(df_pandas.columns) == 1 and inferred_delimiter != ','):
                print(f"DEBUG: Initial read with header for {base_file} yielded single column or was empty. Trying header=None.")
                raise ValueError("Initial read with header failed or yielded single column.")
            else:
                print(f"DEBUG: Read {base_file} with header=True. Rows: {len(df_pandas)}. Columns: {list(df_pandas.columns)}")

        except Exception as e:
            print(f"DEBUG: Attempting read with header=None for {base_file} due to: {e}")
            try:
                if expected_cols:
                    df_pandas = pd.read_csv(file_path, sep=inferred_delimiter, header=None, names=expected_cols, engine='python', on_bad_lines='skip')
                else:
                    df_pandas = pd.read_csv(file_path, sep=inferred_delimiter, header=None, engine='python', on_bad_lines='skip')
                
                if df_pandas.empty:
                    raise ValueError("Read with header=None yielded empty DataFrame.")
                print(f"DEBUG: Read {base_file} with header=None. Rows: {len(df_pandas)}. Columns: {list(df_pandas.columns)}")

            except Exception as e_inner:
                print(f"❌ Error reading {file_path} with inferred delimiter '{inferred_delimiter}' (header=None attempt): {e_inner}. Skipping.")
                return None

        initial_rows = len(df_pandas)
        df_pandas.dropna(how='all', inplace=True)
        if len(df_pandas) < initial_rows:
            print(f"DEBUG: Dropped {initial_rows - len(df_pandas)} all-NA rows from {base_file}.")
        if df_pandas.empty:
            print(f"DEBUG: Skipping {file_path} - DataFrame became empty after dropping all-NA rows.")
            return None

        # --- Column Naming and Timestamp Processing ---
        base_file_name_only = os.path.basename(file_path)
        
        if expected_cols:
            if len(df_pandas.columns) == len(expected_cols):
                df_pandas.columns = expected_cols
                timestamp_col_name = expected_cols[0]
                print(f"DEBUG: Applied expected columns for {base_file_name_only}.")
            else:
                print(f"⚠️ Column count mismatch for {base_file_name_only}. Expected {len(expected_cols)}, got {len(df_pandas.columns)}. Attempting to use first column as timestamp.")
                timestamp_col_name = df_pandas.columns[0]
        else:
            if df_pandas.columns[0].lower().startswith("timestamp"):
                timestamp_col_name = df_pandas.columns[0]
                print(f"DEBUG: Using existing timestamp-like column '{timestamp_col_name}' for {base_file_name_only}.")
            else:
                df_pandas.rename(columns={df_pandas.columns[0]: "timestamp"}, inplace=True)
                timestamp_col_name = "timestamp"
                print(f"DEBUG: Renamed first column to 'timestamp' for {base_file_name_only}.")

        if timestamp_col_name not in df_pandas.columns:
            print(f"❌ Timestamp column '{timestamp_col_name}' not found in {file_path} after column processing. Skipping.")
            return None

        rows_before_ts_drop = len(df_pandas)
        try:
            df_pandas['timestamp'] = pd.to_datetime(df_pandas[timestamp_col_name], unit='ms', errors='coerce')
        except ValueError: # Fallback for non-ms timestamps
            df_pandas['timestamp'] = pd.to_datetime(df_pandas[timestamp_col_name], errors='coerce')

        df_pandas.dropna(subset=['timestamp'], inplace=True)
        if len(df_pandas) < rows_before_ts_drop:
            print(f"DEBUG: Dropped {rows_before_ts_drop - len(df_pandas)} rows from {base_file} due to invalid timestamps.")
        if df_pandas.empty:
            print(f"DEBUG: Skipping {file_path} - DataFrame became empty after dropping rows with invalid timestamps.")
            return None

        if timestamp_col_name != 'timestamp':
            if 'timestamp' in df_pandas.columns and timestamp_col_name in df_pandas.columns:
                df_pandas.drop(columns=[timestamp_col_name], inplace=True)
                print(f"DEBUG: Dropped original timestamp_col_name '{timestamp_col_name}' from {base_file}.")
            elif timestamp_col_name in df_pandas.columns:
                df_pandas.rename(columns={timestamp_col_name: 'timestamp'}, inplace=True)
                print(f"DEBUG: Renamed '{timestamp_col_name}' to 'timestamp' for {base_file}.")

        df_pandas.set_index('timestamp', inplace=True)
        print(f"DEBUG: Set 'timestamp' as index for {base_file}. Remaining rows: {len(df_pandas)}")

        rel_name = relative_name.replace(".csv", "").replace(os.sep, "_")
        full_prefix = f"{task}_{rel_name}" if task else rel_name
        
        new_columns = []
        for col in df_pandas.columns:
            if col not in ['user', 'session']:
                new_columns.append(f"{full_prefix}_{col}")
            else:
                new_columns.append(col)
        df_pandas.columns = new_columns
        print(f"DEBUG: Prefixed columns for {base_file}. New columns: {list(df_pandas.columns)}")

        df_pandas['user'] = user
        df_pandas['session'] = session
        
        print(f"DEBUG: Successfully processed {file_path}. Final Pandas rows: {len(df_pandas)}. Converted to Dask DataFrame.")
        return dd.from_pandas(df_pandas, npartitions=1)

    except Exception as e:
        print(f"❌ Critical error processing {file_path}: {e}")
        return None

# Traverse users, process their data, and save per-user using Dask
print("Starting per-user data processing and saving with Dask (no deduplication)...")
for user in os.listdir(root_dir):
    user_path = os.path.join(root_dir, user)
    if not os.path.isdir(user_path):
        continue

    print(f"\nProcessing data for user: {user}...")
    user_dask_dataframes = []

    for session in os.listdir(user_path):
        session_path = os.path.join(user_path, session)
        if not os.path.isdir(session_path) or not session.lower().startswith('sesion'):
            continue

        for root, _, files in os.walk(session_path):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, session_path)
                    parts = rel_path.split(os.sep)

                    if len(parts) > 1:
                        task = parts[0]
                        rel_name = os.path.join(*parts[1:])
                    else:
                        task = None
                        rel_name = file

                    ddf = process_csv_to_dask_df(file_path, user, session, task, rel_name)
                    if ddf is not None:
                        user_dask_dataframes.append(ddf)
                    else:
                        print(f"DEBUG: No Dask DataFrame generated for {file_path}. It might have been skipped entirely.")


    if user_dask_dataframes:
        print(f"📦 Concatenating {len(user_dask_dataframes)} Dask DataFrames for user {user}...")
        merged_user_ddf = dd.concat(user_dask_dataframes, axis=0, join='outer')

        # 'timestamp' becomes a regular column after reset_index
        merged_user_ddf = merged_user_ddf.reset_index()

        group_cols = ['user', 'session', 'timestamp']

        print(f"DEBUG: Sorting Dask DataFrame for user {user}...")
        # Sort values to ensure correct order, especially for duplicate timestamps
        merged_user_ddf = merged_user_ddf.sort_values(by=group_cols)
        
        # --- REMOVED: merged_user_ddf = merged_user_ddf.drop_duplicates(subset=group_cols, keep='first') ---
        # This line has been removed to keep all rows, even with duplicate timestamps.
        print(f"DEBUG: Deduplication step skipped as requested for user {user}.")

        # To get the columns for reordering from a Dask DataFrame, you need to trigger a small computation
        try:
            cols = merged_user_ddf._meta.columns.tolist()
        except AttributeError:
            print(f"DEBUG: Could not get columns from _meta, computing a head to infer columns for user {user}.")
            cols = merged_user_ddf.head(1).columns.tolist()

        ordered_cols = [col for col in group_cols if col in cols] + \
                       [col for col in cols if col not in group_cols]
        merged_user_ddf = merged_user_ddf[ordered_cols]
        
        output_file_path = os.path.join(output_dir, f"user_{user}_merged.csv")
        
        print(f"DEBUG: Starting final compute and save for user {user} to {output_file_path}...")
        merged_user_ddf.to_csv(output_file_path, index=False, single_file=True)
        print(f"✅ Saved: {output_file_path}")
    else:
        print(f"⚠️ No usable data found for user: {user}")

print("\n🎉 All user data processed and saved with Dask (no deduplication).")
print(f"Merged CSV files are located in the '{output_dir}' directory.")