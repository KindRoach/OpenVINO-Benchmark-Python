import cv2

IMG_SIZE = 480


def read_frames(video_path: str, n_frame: int):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()

    i = 0
    while i < n_frame:
        success, frame = cap.read()
        if success:
            i += 1
            yield frame
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.release()
