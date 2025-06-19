import csv
import pandas as pd

df = pd.read_csv("~/repos/CanaraHack/HuMI/HuMI/000/Sesion1/FINGER_0/SENSORS/sensor_gyro.csv", sep=" ",header=None)
df.columns= ["timestamp","event_type","x","y","z"]

print(df.head())


