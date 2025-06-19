import json
import pandas as pd
import re

def fix_invalid_json(text):
    text = re.sub(r'(\w+)=', r'"\1":', text)            
    text = text.replace("'", '"')                       
    text = re.sub(r':\s*(?=,|\})', ': ""', text)        
    if not text.strip().endswith("}"):
        text += "}" 
    return text

for i in range(599):
    print(str(i))
    if i<10:
        path = "/home/krish/repos/CanaraHack/HuMI (1)/HuMI/00"+str(i)+"/info.json"
    elif i<100:
        path = "/home/krish/repos/CanaraHack/HuMI (1)/HuMI/0"+str(i)+"/info.json"
    else:
        path = "/home/krish/repos/CanaraHack/HuMI (1)/HuMI/"+str(i)+"/info.json"

    with open(path, "r") as f:
        raw = f.read()

    fixed = fix_invalid_json(raw)
    data = json.loads(fixed)

    df = pd.json_normalize([data])
    print(df.head())

    df.to_json(path)
