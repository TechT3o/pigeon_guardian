"""
Run this script to start the annotation with the .mp4 videos stored in pigeon_vids.
Labels in YOLO format are generated in yolo_labels folder.
Press y key to save a datapoint, any other key skips the datapoint.
You can stop the annotation and the next time that you run the script it will continue from the
last annotated frame
"""

from annotator import Annotator


if __name__ == "__main__":

    # LABEL_PATH = ""
    DATA_PATH = "pigeon_vids"
    FRAME_SKIP = 8

    annotator = Annotator(DATA_PATH, frame_skip=FRAME_SKIP)
    annotator.run()
