import numpy as np
from typing import List
from encoder.data_objects.speaker import Speaker
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.nn as nn
import torch

class SpeakerBatch:
    def __init__(self, speakers: List[Speaker], utterances_per_speaker: int, n_frames: int):
        self.speakers = speakers
        self.partials = {s: s.random_partial(utterances_per_speaker, n_frames) for s in speakers}
        
        # Define the target size to be exactly 40 mel channels
        self.target_size = 40
        
        # Prepare the data with fixed mel channel size
        self.data = self.prepare_data()
    
    def prepare_data(self):
        """
        Prepares and pads/truncates the batch data to ensure consistent input size of 40 mel channels.
        """
        batch_data = []
        for s in self.speakers:
            for _, frames, _ in self.partials[s]:
                # Ensure the frames have exactly 40 mel channels
                resized_frame = self.resize_frame(frames, self.target_size)
                
                # Convert to torch tensor
                resized_frame = torch.tensor(resized_frame, dtype=torch.float32).detach()
                batch_data.append(resized_frame)
        
        # Stack into a single tensor of shape (batch_size, n_frames, mel_n)
        return torch.stack(batch_data)

    def resize_frame(self, frame, target_size):
        """
        Resizes the given frame to have exactly target_size mel channels.
        """
        current_size = frame.shape[1]
        if current_size < target_size:
            # Pad with zeros if fewer than target_size mel channels
            return np.pad(frame, ((0, 0), (0, target_size - current_size)), mode='constant', constant_values=0)
        elif current_size > target_size:
            # Truncate if more than target_size mel channels
            return frame[:, :target_size]
        return frame