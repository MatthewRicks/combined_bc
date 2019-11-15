import cv2
import numpy as np
from queue import Queue
from threading import Thread


class VideoCapture:
    # Adapted from https://stackoverflow.com/questions/43665208/how-to-get-the-latest-frame-from-capture-device-camera-in-opencv-python
    def __init__(self, video_device, cam_dim=(256,256)):
        self.video_device = video_device
        self.cam_dim = cam_dim
        self.getting_queue = Queue()
        self.queue = Queue()

        self.getting_thread = Thread(target=self._run_getter)
        self.getting_thread.daemon = True
        self.getting_thread.start()

        self.processing_thread = Thread(target=self._run_processor)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _process_image(self, img):
        h, w, _ = img.shape
        new_w = int( float(w) / float(h) * self.cam_dim[0])
        new_shape = (new_w, self.cam_dim[0])

        resized = cv2.resize(img, new_shape, cv2.INTER_AREA)

        center = new_w // 2
        left = center - self.cam_dim[1] // 2 # assume for now that this is multiple of 2
        right = center + self.cam_dim[1] // 2

        img = resized[:,left:right,:]
        img = np.array([img])
        img = np.transpose(img, (0, 3, 1, 2))
        return img

    def _run_processor(self):
        while True:
            if not self.getting_queue.empty():
                try:
                    frame = self.getting_queue.get_nowait()
                    frame = self._process_image(frame)

                    if not self.queue.empty():
                        try:
                            self.queue.get_nowait()
                        except:
                            pass
                    self.queue.put(frame)
                except:
                    pass

    def _run_getter(self):
        cap = cv2.VideoCapture(self.video_device)

        while True:
            ret, frame = cap.read()

            if not ret:
                print('Camera is not open')
                break

            if not self.getting_queue.empty():
                try:
                    self.getting_queue.get_nowait()
                except:
                    pass

            self.getting_queue.put(frame)

    def read(self):
        return 1, self.queue.get()
