import os
import pandas as pd
import glob

# Set this to your HuMI dataset root
root_dir = "/Users/kkartikaggarwal/Desktop/coding folder2/canrabank/HuMI"
output_dir = "/Users/kkartikaggarwal/Desktop/coding folder2/canrabank/temp_outputs"
os.makedirs(output_dir, exist_ok=True)

column_mappings = {
    "bluetooth.csv": ["timestamp", "bt_name", "bt_mac"],
    "gps.csv": ["timestamp", "orientation", "latitude", "longitude", "altitude", "bearing", "accuracy"],
    "wifi.csv": ["timestamp", "wifi_name", "wifi_level", "wifi_info", "wifi_channel", "wifi_freq"],
    "sensor_gyro.csv": ["timestamp", "orientation", "gyro_x", "gyro_y", "gyro_z"],
    "sensor_grav.csv": ["timestamp", "orientation", "gravity"],
    "sensor_humd.csv": ["timestamp", "orientation", "humidity"],
    "sensor_lacc.csv": ["timestamp", "orientation", "lacc_x", "lacc_y", "lacc_z"],
    "sensor_ligh.csv": ["timestamp", "orientation", "light_level"],
    "sensor_magn.csv": ["timestamp", "orientation", "magn_x", "magn_y", "magn_z"],
    "sensor_nacc.csv": ["timestamp", "orientation", "nacc_x", "nacc_y", "nacc_z"],
    "sensor_prox.csv": ["timestamp", "orientation", "proximity"],
    "sensor_temp.csv": ["timestamp", "orientation", "temperature"],
    "key_data.csv": ["timestamp", "key_field", "ascii_code"],
    "swipe.csv": ["timestamp", "orientation", "x", "y", "pressure", "action"],
    "scroll_X_touch.csv": ["timestamp", "orientation", "x", "y", "pressure", "action"],
    "touch_touch.csv": ["timestamp", "orientation", "x", "y", "pressure", "action"],
    "f_X_touch.csv": ["timestamp", "orientation", "x", "y", "pressure", "action"],
}

def process_csv(file_path, user, session, task, relative_name):
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"⚠️ Skipped empty file: {file_path}")
            return None

        df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
        if df.empty:
            return None

        df.dropna(how='all', inplace=True)
        if df.empty:
            return None

        base_file = os.path.basename(file_path)
        if base_file in column_mappings and len(df.columns) == len(column_mappings[base_file]):
            df.columns = column_mappings[base_file]
        else:
            df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        if df.empty:
            return None

        df.set_index('timestamp', inplace=True)

        rel_name = relative_name.replace(".csv", "").replace(os.sep, "_")
        full_prefix = f"{task}_{rel_name}" if task else rel_name
        df.columns = [f"{full_prefix}_{col}" for col in df.columns]

        df['user'] = user
        df['session'] = session

        return df

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return None

# Traverse the dataset
for user in os.listdir(root_dir):
    user_path = os.path.join(root_dir, user)
    if not os.path.isdir(user_path):
        continue

    for session in os.listdir(user_path):
        session_path = os.path.join(user_path, session)
        if not os.path.isdir(session_path) or not session.lower().startswith('sesion'):
            continue

        dfs = []
        for root, _, files in os.walk(session_path):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, session_path)
                    parts = rel_path.split(os.sep)
                    task = parts[0] if len(parts) > 1 else None
                    rel_name = os.path.join(*parts[1:]) if task else file

                    df = process_csv(file_path, user, session, task, rel_name)
                    if df is not None:
                        dfs.append(df)

        if dfs:
            print(f"💾 Saving merged session: {user} {session}")
            merged = pd.concat(dfs, axis=1)
            merged.reset_index(inplace=True)
            merged.insert(0, "session", session)
            merged.insert(0, "user", user)
            merged.to_csv(f"{output_dir}/{user}_{session}.csv", index=False)

# Final merge of all session CSVs
print("📦 Final merging all session files...")
csvs = glob.glob(f"{output_dir}/*.csv")
final_df = pd.concat([pd.read_csv(f) for f in csvs], axis=0)
final_df.to_csv("final_HUMIdb_merged_cleaned.csv", index=False)
print("✅ Saved: final_HUMIdb_merged_cleaned.csv")
