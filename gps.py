import os
import csv
import ast

base_path = '/Users/riyamehdiratta/Downloads/HuMI '
start_folder = 0
end_folder = 598

def change_delimiter_space_to_comma(file_path):
    try:
        with open(file_path, 'r') as infile:
            lines = infile.readlines()

        updated_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Header:"):
                header_part = line[len("Header:"):].strip()
                try:
                    row = ast.literal_eval(header_part)
                except Exception:
                    print(f"⚠️ Skipped malformed header: {line}")
                    continue
            else:
                try:
                    row = ast.literal_eval(line)
                except Exception:
                    print(f"⚠️ Could not parse line: {line}")
                    continue

            cleaned_row = []
            for item in row:
                item = str(item).strip().replace('"', '').replace("'", "")
                try:
                    if '.' in item:
                        cleaned_row.append(str(float(item)))
                    else:
                        cleaned_row.append(str(int(item)))
                except ValueError:
                    cleaned_row.append(item)

            updated_lines.append(','.join(cleaned_row))

        with open(file_path, 'w') as outfile:
            outfile.write('\n'.join(updated_lines))

        print(f"✅ Cleaned and saved: {file_path}")

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except Exception as e:
        print(f"❌ Error: {e}")

# Traverse folders 000 to 598
for i in range(start_folder, end_folder + 1):
    folder_name = f"{i:03d}"
    folder_path = os.path.join(base_path, folder_name)

    if not os.path.isdir(folder_path):
        continue

    for session in os.listdir(folder_path):
        session_path = os.path.join(folder_path, session)

        if not os.path.isdir(session_path):
            continue

        gps_file = os.path.join(session_path, 'gps.csv')

        if os.path.isfile(gps_file):
            change_delimiter_space_to_comma(gps_file)
