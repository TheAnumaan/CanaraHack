import os
import pandas as pd

root_dir = os.path.expanduser("/Users/kkartikaggarwal/REPOS/CanaraHack/HuMI/")

keywords = ["swipe", "touch", "SCROLLUP", "SCROLLDOWN", "touch_touch"]

def has_header(row):
    # Simple heuristic: if any expected keyword is present in first row, it's likely a header
    header_keywords = ["timestamp", "orientation", "x", "y", "z", "SSID", "MAC", "altitude", "accuracy"]
    return any(kw.lower() in str(row).lower() for kw in header_keywords)

for dirpath, _, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith(".csv"):
            file_path = os.path.join(dirpath, file)

            try:
                # Read first row to check if it's a header
                with open(file_path, 'r') as f:
                    first_line = f.readline()

                if has_header(first_line):
                    df = pd.read_csv(file_path)  # Already has header
                else:
                    df = pd.read_csv(file_path, header=None)  # No header

                    # Assign headers based on file type
                    if any(kw in file_path.lower() for kw in keywords):
                        df.columns = ["timestamp(ms)", "orientation", "x", "y", "p", "action"]

                    elif "gyro" in file_path.lower() or \
                         "lacc" in file_path.lower() or \
                         "magn" in file_path.lower() or \
                         "nacc" in file_path.lower():
                        df.columns = ["timestamp(ms)", "orientation", "x", "y", "z"]

                    elif "grav" in file_path.lower():
                        df.columns = ["timestamp(ms)", "orientation", "gravity_data"]

                    elif "ligh" in file_path.lower():
                        df.columns = ["timestamp(ms)", "orientation", "light_data"]

                    elif "prox" in file_path.lower():
                        df.columns = ["timestamp(ms)", "orientation", "proximity"]

                    elif "temp" in file_path.lower():
                        df.columns = ["timestamp(ms)", "orientation", "temperature"]

                    elif "wifi" in file_path.lower():
                        df.columns = ["timestamp(ms)", "SSID", "level", "info", "channel", "frequency"]

                    elif "bluetooth" in file_path.lower():
                        df.columns = ["timestamp(ms)", "name", "MAC"]

                    elif "gps" in file_path.lower():
                        df.columns = ["timestamp(ms)", "orientation", "latitude", "longitude", "altitude", "bearing", "accuracy"]

                    else:
                        print(f"Unrecognized sensor file format: {file_path}")
                        continue

                print(f"Processed: {file_path}")
                print(df.head())

                df.to_csv(file_path, sep=",", index=False)

            except Exception as e:
                print(f"❌ Failed: {file_path} → {e}")
