import torch
from torch import nn


class BiLSTM(nn.Module):
    inference_chunk_length = 512

    def __init__(self, input_features, recurrent_features):
        super().__init__()
        #sgu: hidden_size = recurrent_features
        # since batch_first = True, input and output shape is (batch, seq, feature)
        self.rnn = nn.LSTM(input_features, recurrent_features, batch_first=True, bidirectional=True) 

    def forward(self, x, need_states=False):
        """
        sgu:
        x: shape (batch, seq, model_size) where model_size = model_complexity * 16 
        """
        if self.training:
            y, (h, c) = self.rnn(x)
            if need_states:
                return y, (h,c)
            else:
                return y
        else:
            # evaluation mode: support for longer sequences that do not fit in memory
            batch_size, sequence_length, input_features = x.shape
            hidden_size = self.rnn.hidden_size
            num_directions = 2 if self.rnn.bidirectional else 1

            h = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            c = torch.zeros(num_directions, batch_size, hidden_size, device=x.device)
            output = torch.zeros(batch_size, sequence_length, num_directions * hidden_size, device=x.device)

            # forward direction
            slices = range(0, sequence_length, self.inference_chunk_length)
            for start in slices:
                end = start + self.inference_chunk_length
                output[:, start:end, :], (h, c) = self.rnn(x[:, start:end, :], (h, c))

            # reverse direction
            if self.rnn.bidirectional:
                #save hidden states of forward direction 
                h_f = h[0,:].clone().detach().unsqueeze(0)
                c_f = c[0,:].clone().detach().unsqueeze(0)
                
                h.zero_()
                c.zero_()

                for start in reversed(slices):
                    end = start + self.inference_chunk_length
                    result, (h, c) = self.rnn(x[:, start:end, :], (h, c))
                    output[:, start:end, hidden_size:] = result[:, :, hidden_size:]

                #save hidden states of backward direction 
                h_b = h[1,:].clone().detach().unsqueeze(0)
                c_b = c[1,:].clone().detach().unsqueeze(0)
                h = torch.cat([h_f, h_b], dim=0)
                c = torch.cat([c_f, c_b], dim=0)
            if need_states:
                return output, (h,c)
            else:
                return output 
