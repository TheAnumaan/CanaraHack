import os
import pandas as pd
from functools import reduce

# Change to the desired user folder (e.g., u00, u01)
target_user = "000"

# Path settings
root_dir = "/home/krish/repos/CanaraHack/HuMI (1)/HuMI/"
output_dir = "/home/krish/repos/CanaraHack/"
os.makedirs(output_dir, exist_ok=True)

# Known column mappings
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
    "scroll_d_touch.csv": ["timestamp", "orientation", "x", "y", "pressure", "action"],
    "scroll_u_touch.csv": ["timestamp", "orientation", "x", "y", "pressure", "action"],
    "touch_touch.csv": ["timestamp", "orientation", "x", "y", "pressure", "action"],
    "f_0_touch.csv": ["timestamp", "orientation", "x", "y", "pressure", "action"],
    # Add more FINGER_X touch mappings if needed
}

# Main CSV parser
def process_csv(file_path, user, session, task, relative_name):
    try:
        print(f"📄 Processing: {file_path}")
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            print(f"⚠️ Skipped empty file: {file_path}")
            return None

        df = pd.read_csv(file_path, engine='python', on_bad_lines='skip')
        if df.empty:
            print(f"⚠️ Skipped empty content in: {file_path}")
            return None

        df.dropna(how='all', inplace=True)
        if df.empty:
            print(f"⚠️ All NaNs in: {file_path}")
            return None

        base_file = os.path.basename(file_path)
        if base_file in column_mappings and len(df.columns) == len(column_mappings[base_file]):
            df.columns = column_mappings[base_file]
        else:
            df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        if df.empty:
            print(f"⚠️ All invalid timestamps in: {file_path}")
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

# Begin per-user merge
user_path = os.path.join(root_dir, target_user)
if not os.path.isdir(user_path):
    print(f"❌ User folder not found: {user_path}")
    exit()

user_dfs = []

for session in os.listdir(user_path):
    session_path = os.path.join(user_path, session)
    if not os.path.isdir(session_path) or not session.lower().startswith("sesion"):
        continue

    dfs = []
    for root, _, files in os.walk(session_path):
        for file in files:
            if not file.endswith(".csv"):
                continue  # Skip non-CSV files like .npy, .3gp

            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, session_path)
            parts = rel_path.split(os.sep)

            # Determine task and relative filename
            if len(parts) >= 2 and parts[1] == "SENSORS":
                task = parts[0]
                rel_name = parts[2] if len(parts) >= 3 else parts[1]
            elif len(parts) >= 2:
                task = parts[0]
                rel_name = os.path.join(*parts[1:])
            else:
                task = None
                rel_name = parts[0]

            print(f"➡️ Detected: task={task}, rel_name={rel_name}")
            df = process_csv(file_path, target_user, session, task, rel_name)
            if df is not None:
                dfs.append(df)

    if dfs:
        print(f"🔗 Merging session: {target_user} {session}")
        dfs = [df.reset_index() for df in dfs]
        merged_session = reduce(
            lambda left, right: pd.merge(left, right, on=["timestamp", "user", "session"], how="outer"),
            dfs
        )
        user_dfs.append(merged_session)

if user_dfs:
    final_user_df = pd.concat(user_dfs, axis=0, ignore_index=True)
    output_path = os.path.join(output_dir, f"{target_user}_all_sessions.csv")
    final_user_df.to_csv(output_path, index=False)
    print(f"✅ Final user CSV saved: {output_path}")
else:
    print(f"⚠️ No valid sessions found for user {target_user}")
