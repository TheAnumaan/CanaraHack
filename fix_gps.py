import os
import pandas as pd

root_dir = os.path.expanduser("/Users/kkartikaggarwal/Desktop/coding folder2/canrabank/HuMI/")

for dirpath, _, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith(".csv"):
            file_path = os.path.join(dirpath, file)

            try:
                df = pd.read_csv(file_path, sep=",", header=None)


                if "gps" in file_path.lower():
                    df.columns = ["timestamp(ms)","orientation","latitude","longitude","altitude","bearing","accuracy"]
                else:
                    continue

                print(f"Processed: {file_path}")
                print(df.head())

                df.to_csv(file_path, sep=",", index=False)

            except Exception as e:
                print(f"Failed: {file_path} → {e}")