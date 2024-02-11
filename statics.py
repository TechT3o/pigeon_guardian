import os
from typing import Tuple, List


def check_make_dir(path: str) -> bool:
    """
    Checks if directory exists and if it doesn't it creates it
    :param path: Path of directory
    :return: Boolean that shows that whether directory existed
    """
    exists = os.path.exists(path)
    if not exists:
        os.makedirs(path)
    return exists


def get_progress(data_path: str, label_path: str) -> Tuple[List[str], int]:
    """
    Find remaining videos and the last annotated frame
    :param data_path: Path where data video files are found
    :param label_path: Path where the annotations should go
    :return: Tuple with a list of the names of the remaining videos and the number of the last annotated frame
    """

    # Create label_path if it doesn't already exist
    assert check_make_dir(data_path)

    check_make_dir(label_path)
    assert (len(os.listdir(data_path)) > 0), "No video data found to label"

    if len(os.listdir(label_path)) == 0:
        print("No annotations yet, starting from zero")
        return sorted(os.listdir(data_path)), 0

    # Find existing videos and annotated videos and use that to find remaining videos
    remaining_videos = []
    video_list = [video_name.split(".")[0] for video_name in sorted(os.listdir(data_path))]
    annotated_videos = list(set([annotated_vid.split("_")[0] for annotated_vid in
                                 sorted(os.listdir(label_path)) if ".jpg" in annotated_vid]))

    for video in video_list:
        if video not in annotated_videos[:-1]:
            remaining_videos.append(video + ".mp4")

    # Gets last annotated frame from last video
    last_frame = sorted([int(annotated_vid.split("_")[1].split(".")[0]) for annotated_vid in
                         sorted(os.listdir(label_path)) if (".jpg" in annotated_vid) and
                         (annotated_vid.split("_")[0] == annotated_videos[-1])])[-1]
    print(f"Current_progress video: {remaining_videos[0]} on frame {last_frame}")

    return remaining_videos, last_frame
