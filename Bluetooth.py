import os

base_path = '/home/krish/repos/CanaraHack/HuMI/HuMI'
start_folder = 0
end_folder = 598

def process_bluetooth_file(file_path):
    try:
        # Read original content
        with open(file_path, 'r') as infile:
            lines = infile.readlines()

        processed_lines = []

        for line in lines:
            line = line.strip()
            first_space = line.find(' ')
            last_space = line.rfind(' ')

            if first_space == -1 or last_space == -1 or first_space == last_space:
                processed_lines.append(line)
                continue

            part1 = line[:first_space]
            part2 = line[first_space + 1:last_space]
            part3 = line[last_space + 1:]

            new_line = f"{part1},{part2},{part3}"
            processed_lines.append(new_line)

        # Write back to the same file
        with open(file_path, 'w') as outfile:
            for line in processed_lines:
                outfile.write(line + '\n')

        print(f"✅ Overwritten: {file_path}")

    except Exception as e:
        print(f"❌ Failed to process {file_path}: {e}")

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

        bluetooth_file = os.path.join(session_path, 'bluetooth.csv')

        if os.path.isfile(bluetooth_file):
            process_bluetooth_file(bluetooth_file)
