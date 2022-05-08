
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.models.attentions.util import *
from src.models.embeddings.util import *
from src.data_helpers.vocab import Vocab, device



class Transformer(nn.Module):

    def __init__(self, vocab: Vocab,
                 args):
        """

        :param vocab: Vocab
            The vocabulary normally built on the training data
        :param args:
            mode: rand/static/non-static/multichannel the mode of initialising embeddings
            hidden_size: (int) The size of the hidden layer
            n_layers: (int) The number of hidden layers
            bidirectional: (bool) Whether or not using bidirectional connection
            dropout: (float) The dropout parameter for RNN (GRU or LSTM)
        """

        super(Transformer, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.args = args
        self.use_last_hidden_state = args.use_last_hidden_state

        self.hidden_size = args.hidden_size

        self.attention_mode = args.attention_mode
        self.output_size = self.hidden_size

        self.dropout = args.dropout
        self.embedding = init_embedding_layer(args, vocab)
        self.transformer = nn.Transformer(self.embedding.output_size)

        if self.rnn_model.lower() == "gru":
            self.rnn = nn.GRU(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                              bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)
        else:
            self.rnn = nn.LSTM(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                               bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)

        self.use_dropout = args.dropout > 0
        self.dropout = nn.Dropout(args.dropout)
        init_attention_layer(self)