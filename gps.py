import os

# === CONFIGURATION ===
base_path = '/home/krish/repos/CanaraHack/HuMI/HuMI'
start_folder = 0
end_folder = 598

# === CLEANER FUNCTION ===
def change_delimiter_space_to_comma(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as infile:
            lines = infile.readlines()

        updated_lines = []

        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue  # skip blank lines

            # Use whitespace as delimiter
            items = line.split()

            cleaned_row = []
            for item in items:
                item = item.replace('"', '').replace("'", "")
                try:
                    # Try to parse as float or int
                    if '.' in item:
                        cleaned_row.append(str(float(item)))
                    else:
                        cleaned_row.append(str(int(item)))
                except ValueError:
                    cleaned_row.append(item)  # fallback: leave as string

            updated_lines.append(','.join(cleaned_row))

        # Overwrite the original file
        with open(file_path, 'w', encoding='utf-8') as outfile:
            outfile.write('\n'.join(updated_lines))

        print(f"✅ Cleaned and saved: {file_path}")

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

# === DIRECTORY TRAVERSAL ===
def traverse_and_clean():
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

# === RUN SCRIPT ===
if __name__ == "__main__":
    traverse_and_clean()
