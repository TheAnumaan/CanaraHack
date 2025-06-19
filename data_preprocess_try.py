import json
import numpy as np
import pandas as pd

with open("/home/krish/repos/CanaraHack/HuMI/HuMI/000/info.json",'r') as file:
    data = json.load(file)

print(data)

print()
print()
print()

df = pd.json_normalize(data)
print(df)
