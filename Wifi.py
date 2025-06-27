import os

base_path = '/Users/kkartikaggarwal/REPOS/CanaraHack/HuMI/'
start_folder = 0
end_folder = 598

def clean_wifi_csv(file_path):
    try:
        cleaned_lines = []

        with open(file_path, 'r') as infile:
            for line in infile:
                # Split line using any whitespace
                columns = line.strip().split()

                # Only include rows with exactly 6 columns
                if len(columns) == 6:
                    cleaned_lines.append(','.join(columns))

        with open(file_path, 'w') as outfile:
            for line in cleaned_lines:
                outfile.write(line + '\n')

        print(f"✅ Cleaned and formatted: {file_path}")

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

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

        wifi_file = os.path.join(session_path, 'wifi.csv')

        if os.path.isfile(wifi_file):
            clean_wifi_csv(wifi_file)

# Optional: Run on a single file for testing
test_file = 'wifi.csv'
if os.path.exists(test_file):
    clean_wifi_csv(test_file)
