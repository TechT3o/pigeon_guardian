import os

def convert_class_to_int(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
            with open(filepath, 'w+') as file:
                for line in lines:
                    parts = line.split(" ")
                    if len(parts) == 5:  # Ensure it's a valid annotation line
                        # Convert class index from float to int
                        parts[0] = str(int(float(parts[0])))
                        print(filename, ' '.join(parts))
                        file.write(' '.join(parts))

if __name__ == "__main__":
    convert_class_to_int('yolo_labels')