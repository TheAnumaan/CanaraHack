import os
import pandas as pd

root_dir = "~/repos/CanaraHack/HuMI/HuMI/"

for dirpath,_, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith(".csv"):
            file_path = os.path.join(dirpath, file)
            df = pd.read_csv(file_path, sep=" ",header=None)
            if "gyro" in file_path:
                df.columns=["timestamp(ms)","orientation","x","y","z"]
                print(df.head())
                df.to_csv(file_path,sep=",",index=False)
            elif "grav" in file_path:
                df.columns=["timestamp(ms)","orientation","gravity_data"]
                print(df.head())
                df.to_csv(file_path,sep=",",index=False)
            elif "lacc" in file_path:
                df.columns=["timestamp(ms)","orientation","x","y","z"]
                print(df.head())
                df.to_csv(file_path,sep=",",index=False)
            elif "ligh" in file_path:
                df.columns=["timestamp(ms)","orientation","light_data"]
                print(df.head())
                df.to_csv(file_path,sep=",",index=False)
            elif "magn" in file_path:
                df.columns=["timestamp(ms)","orientation","x","y","z"]
                print(df.head())
                df.to_csv(file_path,sep=",",index=False)
            elif "nacc" in file_path:
                df.columns=["timestamp(ms)","orientation","x","y","z"]
                print(df.head())
                df.to_csv(file_path,sep=",",index=False)
            elif "prox" in file_path:
                df.columns=["timestamp(ms)","orientation","proximity"]
                print(df.head())
                df.to_csv(file_path,sep=",",index=False)
            elif "temp" in file_path:
                df.columns=["timestamp(ms)","orientation","temperature"]
                print(df.head())
                df.to_csv(file_path,sep=",",index=False)
            else:
                continue

#path = "~/repos/CanaraHack/HuMI/HuMI/000/Sesion1/FINGER_0/SENSORS/sensor_gyro.csv"


