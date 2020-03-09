"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

import math
import torch
import torch.nn.functional as F
from torch import nn

from .lstm import BiLSTM
from .mel import melspectrogram
from .transformer_amt import Transformer_AMT

class ConvStack(nn.Module):
    """
    This is the acoustic model in the onset and frame paper 
    """
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2)) #1 is the channel
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=2500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(100000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # (seq_len, d_model)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class OnsetsAndFrames(nn.Module):
    #output_feature is MAX_MIDI-MIN_MIDI +1 =88
    #input_features = N_MEL = 229 by default 
    def __init__(self, input_features, output_features, model_complexity=48):
        super().__init__()

        model_size = model_complexity * 16

        #sgu: output_size is half so that two direction together output_size 
        #sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)
        sequence_model = lambda input_size: Transformer_AMT(input_size,
                                num_encoder_layers=2, dim_feedforward=input_size)
        self.pos_enc = PositionalEncoding(input_features)
        self.onset_stack = nn.Sequential(
            #ConvStack(input_features, model_size),
            #sequence_model(model_size, model_size),
            sequence_model(input_features),
            nn.Linear(input_features, output_features),
            nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            #ConvStack(input_features, model_size),
            #sequence_model(model_size, model_size),
            sequence_model(input_features),
            nn.Linear(input_features, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            #ConvStack(input_features, model_size),
            #Transformer_AMT(input_features),
            sequence_model(input_features),
            nn.Linear(input_features, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            #sequence_model(output_features * 3, model_size),
            sequence_model(3 * output_features),
            nn.Linear(3 * output_features, output_features),
            nn.Sigmoid()
        )
        self.velocity_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features)
        )

    def forward(self, mel):
        x = self.pos_enc(mel)
        onset_pred = self.onset_stack(x)
        offset_pred = self.offset_stack(x)
        activation_pred = self.frame_stack(x)
        # sgu: why to use detach()?
        # tensor.detach() :  creates a tensor that shares storage with tensor that does not require grad
        combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred, velocity_pred

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        velocity_label = batch['velocity']

        mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)

        onset_pred, offset_pred, _, frame_pred, velocity_pred = self(mel)

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            'velocity': velocity_pred.reshape(*velocity_label.shape)
        }

        losses = {
            'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
            'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
            'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
            'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        }

        return predictions, losses

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator

