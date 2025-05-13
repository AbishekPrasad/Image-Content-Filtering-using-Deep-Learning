def load_image_data_from_multiple_files(file_paths):
    image_data = []
    for file_path in file_paths:
        print(f"Opening file: {file_path}")
        try:
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 2:
                        print(f"Skipping malformed line: {line}")
                        continue

                    url, label = parts
                    try:
                        label = int(label)
                        image_data.append({"url": url, "label": label})
                    except ValueError:
                        print(f"Invalid label: {label}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
    print(f"Total images loaded: {len(image_data)}")
    return image_data
