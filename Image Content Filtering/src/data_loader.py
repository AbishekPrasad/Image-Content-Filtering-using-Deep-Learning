import random

def load_data_from_files(class_files):
    class_to_label = {cls: i for i, cls in enumerate(class_files.keys())}
    data = []
    for cls, filename in class_files.items():
        with open(filename, 'r') as f:
            urls = f.read().splitlines()
            for url in urls:
                data.append((url, class_to_label[cls]))
    random.shuffle(data)
    return data, class_to_label
