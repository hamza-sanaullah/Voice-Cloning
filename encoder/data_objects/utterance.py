import numpy as np


class Utterance:
    def __init__(self, frames_fpath, wave_fpath):
        self.frames_fpath = frames_fpath
        self.wave_fpath = wave_fpath
        
    def get_frames(self):
        return np.load(self.frames_fpath)

    def random_partial(self, n_frames):
        """
        Crops the frames into a partial utterance of n_frames
        
        :param n_frames: The number of frames of the partial utterance
        :return: the partial utterance frames and a tuple indicating the start and end of the 
        partial utterance in the complete utterance.
        """
        frames = self.get_frames()

        # If frames are too short, pad them with zeros
        if frames.shape[0] < n_frames:
            padding = np.zeros((n_frames - frames.shape[0], frames.shape[1]))
            frames = np.vstack((frames, padding))
        else:
            # Randomly select a segment if frames are longer
            start = np.random.randint(0, frames.shape[0] - n_frames + 1)
            frames = frames[start:start + n_frames]

        return frames, (0, n_frames)