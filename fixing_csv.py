import os
import pandas as pd

root_dir = os.path.expanduser("/Users/kkartikaggarwal/REPOS/CanaraHack/HuMI/")

keywords = ["swipe", "touch", "scroll", "touch_touch"]

for dirpath, _, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith(".csv"):
            file_path = os.path.join(dirpath, file)

            try:
                df = pd.read_csv(file_path, sep=" ", header=None)

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
                    df.columns = ["timestamp(ms)", "SSID","level","info","channel","frequency"]
                elif "bluetooth" in file_path.lower():
                    df.columns = ["timestamp (ms)","name","MAC"]
                else:
                    print(f"Unrecognized sensor file format: {file_path}")
                    continue

                print(f"Processed: {file_path}")
                print(df.head())

                df.to_csv(file_path, sep=",", index=False)

            except Exception as e:
                print(f"Failed: {file_path} → {e}")

