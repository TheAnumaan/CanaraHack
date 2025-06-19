import os

base_path = '/home/krish/repos/CanaraHack/HuMI/HuMI'
start_folder = 0
end_folder = 598

def change_delimiter_space_to_comma(file_path):
    try:
        with open(file_path, 'r') as infile:
            lines = infile.readlines()

        # Replace spaces with commas (but only between words)
        updated_lines = [','.join(line.strip().split()) for line in lines]

        with open(file_path, 'w') as outfile:
            for line in updated_lines:
                outfile.write(line + '\n')

        print(f"✅ Delimiter changed to comma in '{file_path}'")

    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

# Traverse folders 000 to 600
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
            change_delimiter_space_to_comma(wifi_file)


# Example usage
file_path = 'wifi.csv'
change_delimiter_space_to_comma(file_path)
