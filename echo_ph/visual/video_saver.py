import cv2
import os


class VideoSaver():
    def __init__(self, video_title, grad_cam_frames, out_dir='vis_videos', max_frames=50, fps=10):
        """
        Class to save spatio-temporal saliency maps as videos.
        :param video_title: Title to save the video
        :param grad_cam_frames: The saliency map sequence
        :param out_dir: Directory to store the video
        :param max_frames: If cut sequence at certain frame number
        :param fps: frames per sec, for how slow or fast the video appears
        """
        self.video_title = str(video_title)
        self.grad_cam_frames = grad_cam_frames
        self.max_frames = max_frames
        self.fps = fps
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def save_video(self):
        size = (self.grad_cam_frames[0].shape[0], self.grad_cam_frames[0].shape[1])
        out_path = os.path.join(self.out_dir, self.video_title + '.avi')
        result = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, size)
        frames = self.grad_cam_frames[0:self.max_frames]  # E.g. only first 50 frames, or approx first 5 heart-beats
        for frame in frames:
            result.write(frame)  # Write the frame into the file .avi file
        result.release()
        cv2.destroyAllWindows()  # Closes all the frames

