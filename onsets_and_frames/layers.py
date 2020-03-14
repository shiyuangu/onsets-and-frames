from torch import nn
from .lstm import BiLSTM

sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

def get_layers(name, **kwargs):
    if name == "original":
        # this is the same as original implementation
        model_size = kwargs['model_size']
        input_features = kwargs['input_features']
        output_features = kwargs['output_features']
        rv =  nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        return rv
    
    elif name == "conv-attn-rnn":
        rv = ConvAttnRnn(**kwargs)
        return rv 
    else:
        raise RuntimeError("name %s not known" % (name,))

class ConvAttnRnn(nn.Module):
    def __init__(self, input_features, output_features, model_size):
        """
        :param input_features: int, app default is N_MELS=88 
        :param output_features: int, app default is MAX_MIDI-MIN_MIDI +1 =88
        :param model_size: int. app default is model_complexity(48) * 16 
        """
        assert model_size % 2 ==0, "model_size must be even number" 
        super().__init__()
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)
        self.conv_stack = ConvStack(input_features, model_size)
        self.attn = nn.MultiheadAttention(model_size, num_heads=1)
        self.rnn = sequence_model(model_size, model_size)
        self.linear = nn.Linear(model_size, output_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, mel):
        """
        :param mel: Tensor of shape (batch_size, seq_len, N_MELS) 
        :return: Tensor of shape (batch_size, seq_len, output_features)
        """
        x = self.conv_stack(mel) #(batch_size, seq_len, model_size)
        x = x.transpose(0,1)     #(seq_len, batch_size, model_size)
        x, _ = self.attn(x,x,x)  # x: (seq_len, batch_size, model_size)
        x = x.transpose(0,1)     #(batch_size, seq_len, model_size)
        x  = self.rnn(x)       #(batch_size, seq_len, model_size
        x = self.linear(x)       #(batch_size, seq_len, output_features)
        x = self.sigmoid(x)
        return x

class ConvStack(nn.Module):
    """
    This is the acoustic model in the onset and frame paper 
    """
    def __init__(self, input_features, output_features):
        #output_feature is MAX_MIDI-MIN_MIDI +1 =88
        #input_features = N_MEL = 229 by default 
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
        """
        :param mel: Tensor of shape (batch_size, seq_len, N_MELS) 
        :return: Tensor of shape (batch_size, seq_len, N_MELS)
        """
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2)) #1 is the channel
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x
        
        
        
