import os.path
import cv2
import ultralytics.engine.results
from statics import get_progress
from typing import Tuple
from yolo_model import YOLOModel


class Annotator:
    def __init__(self, data_path: str, label_path: str = "yolo_labels", frame_skip: int = 4):
        """
        Class constructor.
        :param data_path: Path where the .mp4 data video files are stored
        :param label_path: Path where the annotation labels will be stored
        :param frame_skip: How many frames to skip after each labeling
        """

        # Set frame skip to skip some of the frames of the video,
        # helps get more variety of data and finish annotating a video faster
        self.FRAME_SKIP = frame_skip

        self.data_path = data_path
        self.label_path = label_path

        self.remaining_videos, self.current_frame = get_progress(data_path, label_path)
        self.current_video = self.remaining_videos[0]

        self.video_cap, self.total_frames = self.change_video_cap(self.current_video)
        print(f"Remaining videos: {self.remaining_videos}")
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        # self.frame_width = self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # self.frame_height = self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Initiate yolo model
        self.yolo_model = YOLOModel()

    def change_video_cap(self, path: str) -> Tuple[cv2.VideoCapture, int]:
        """
        Changes video file from which video capture object reads and updates attributes
        related to video
        :param path: Path of new video file
        :return: tuple with new videocapture object and integer frame
        """

        video_cap = cv2.VideoCapture(os.path.join(self.data_path, path))
        self.current_video = self.remaining_videos[0]
        # remove video from remaining ones
        del self.remaining_videos[0]

        length = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_cap = video_cap
        self.total_frames = length
        print(f"Total frames are {self.total_frames} in video {self.current_video}")
        # self.frame_width = self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # self.frame_height = self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        return video_cap, length

    def skip_frame(self) -> None:
        """
        Increments the current frame with the frame skip and changes to the next video
        if the end of the current video has been reached
        :return: None
        """

        self.current_frame += self.FRAME_SKIP
        print(f"Next frame {self.current_frame}", self.total_frames)

        if self.current_frame > self.total_frames:
            try:
                self.change_video_cap(self.remaining_videos[0])
                self.current_frame = 0
            except IndexError:
                print("No videos remaining to annotate")

    @staticmethod
    def write_yolo(path: str, results: ultralytics.engine.results.Results) -> bool:
        """
        Allows user to select which detections to label and writes the .txt file
        in YOLO format
        :param path: path of text file
        :param results: results obtained from yolo model detection
        :return: Boolean value depending on if any datapoints were labeled
        """

        counts = 0
        with open(path, "w+") as yolo_txt:
            # get detection class, img coordinates and normalized box coordinates (required for YOLO training)
            for object_class, xyxy, xywhn in zip(results.boxes.cls, results.boxes.xyxy, results.boxes.xywhn):
                pigeon_img = results.orig_img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]): int(xyxy[2])]
                cv2.imshow("Detection", cv2.resize(pigeon_img, (400, 400)))

                # Press y to accept datapoint
                if cv2.waitKey(0) == ord("y"):
                    counts += 1
                    print(f"Datapoint {counts} accepted")
                    yolo_txt.write(f"{int(object_class)} {xywhn[0]} {xywhn[1]} {xywhn[2]} {xywhn[3]}\n")
                cv2.destroyAllWindows()

        count_check = counts != 0
        if not count_check:
            os.remove(path)
        return count_check

    def run(self) -> None:
        """
        Main loop. Reads frame, performs detection, lets labeling and increments to next frame
        :return: None
        """
        while self.video_cap.isOpened():

            print(f"Current video: {self.current_video} on frame {self.current_frame}")

            ret, frame = self.video_cap.read()
            if not ret:
                print("Couldn't read frame")
                break

            # Change parameters of detection
            # Class 14 is for bird in COCO dataset
            results = self.yolo_model.predict(frame, conf=0.1, classes=14, iou=0.7)

            # handle no returned results
            if len(results[0]) == 0:
                print(f"No results on vid {self.current_video} on frame {self.current_frame}")
                # Set next frame
                self.skip_frame()
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                continue

            frame_name = f"{self.current_video.split('.')[0]}_{self.current_frame}"
            if self.write_yolo(os.path.join(self.label_path, frame_name + ".txt"), results[0]):
                cv2.imwrite(os.path.join(self.label_path, frame_name + ".jpg"), frame)

            # Set next frame
            self.skip_frame()
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

        self.video_cap.release()
