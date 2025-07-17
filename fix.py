import os
import pandas as pd

root_dir = os.path.expanduser("/Users/riyamehdiratta/Desktop/hackathon/HuMI_final")

gesture_folders = ["touch", "scrollup", "scrolldown"]

def has_header(row):
    header_keywords = ["timestamp", "orientation", "x", "y", "z", "SSID", "MAC", "altitude", "accuracy", "action"]
    return any(kw.lower() in str(row).lower() for kw in header_keywords)

for dirpath, _, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith(".csv"):
            file_path = os.path.join(dirpath, file)

            try:
                parent_folder = os.path.basename(os.path.dirname(file_path)).lower()
                grandparent_folder = os.path.basename(os.path.dirname(os.path.dirname(file_path))).lower()

                print(f"\n📂 FILE: {file_path}")
                print(f"    ↪ PARENT: {parent_folder}")
                print(f"    ↪ GRANDPARENT: {grandparent_folder}")

                with open(file_path, 'r') as f:
                    first_line = f.readline()

                if has_header(first_line):
                    df = pd.read_csv(file_path, sep=None, engine="python")
                    print("✅ Header already exists")
                else:
                    df = pd.read_csv(file_path, sep=None, header=None, engine="python")
                    print("⚠️ Header missing — assigning now")

                    expected_cols = None
                    file_lower = file.lower()

                    if "wifi" in file_lower:
                        expected_cols = ["timestamp(ms)", "SSID", "level", "info", "channel", "frequency"]
                    else:
                        continue

                    if expected_cols:
                        if df.shape[1] != len(expected_cols):
                            print(f"❌ Column mismatch: File has {df.shape[1]} cols, expected {len(expected_cols)} → Skipping")
                            continue
                        df.columns = expected_cols
                    else:
                        print("⚠️ Unknown file type → Skipping")
                        continue

                df.to_csv(file_path, sep=",", index=False)
                print(f"✅ Saved: {file_path}")

            except Exception as e:
                print(f"❌ Failed: {file_path} → {e}")

print(os.path.exists("/Users/riyamehdiratta/Desktop/hackathon/HuMI_final"))