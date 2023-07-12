import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import os
import unicodedata
import codecs
import itertools
import re
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
# processing the words

PAD_token = 0
SOS_token = 1
EOS_token = 2


class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD",
                           SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, min_count):
        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(len(keep_words),
              len(self.word2index), len(keep_words) / len(self.word2index)))

        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: 'PAD',
                           SOS_token: 'SOS', EOS_token: 'EOS'}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)


def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


datafile = os.path.join("wData", "formatted_movie_lines.txt")
print("Reading and processing file. . . Please wait")
lines = open(datafile, encoding='utf-8').read().strip().split('\n')
pairs = [[normalizeString(s) for s in pair.split('\t')] for pair in lines]
print("Done Reading!")
voc = Vocabulary("Cornell movie dialog corpus")

MAX_LENGTH = 10


def filterPair(p):
    return len(p[0].split()) < MAX_LENGTH and len(p[1].split()) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


pairs = [pair for pair in pairs if len(pair) > 1]
print(f"There are {len(pairs)} pairs/ conversations in the dataset.")
pairs = filterPairs(pairs)
print(f"After filtering, there are {len(pairs)} pairs/conversations.")


for pair in pairs:
    voc.addSentence(pair[0])
    voc.addSentence(pair[1])
print("Counted words : ", voc.num_words)
# for pair in pairs[:10]:
#     print(pair)

MIN_COUNT = 3


def trimRareWords(voc, pairs, MIN_COUNT):
    voc.trim(MIN_COUNT)
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True

        for word in input_sentence.split(" "):
            if word not in voc.word2index:
                keep_input = False
                break
        for word in output_sentence.split(" "):
            if word not in voc.word2index:
                keep_output = False
                break

        if keep_input and keep_output:
            keep_pairs.append(pair)

    print(
        f"Trimmed from {len(pairs)} pairs to {len(keep_pairs)}, {len(keep_pairs)/ len(pairs)} of total.")
    return keep_pairs


pairs = trimRareWords(voc, pairs, MIN_COUNT)


def indexesFromSentences(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


inp = []
out = []
i = 0
for pair in pairs[:10]:
    inp.append(pair[0])
    out.append(pair[1])
# print(inp)
# print(len(inp))
indexes = [indexesFromSentences(voc, sentence) for sentence in inp]

# print(indexes)


def zeroPadding(l, fillvalue=0):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


leng = [len(ind) for ind in indexes]

test_result = zeroPadding(indexes)


def binaryMatrix(l, value=0):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


binary_result = binaryMatrix(test_result)
# print(binary_result)


def inputVar(l, voc):
    indexes_batch = [indexesFromSentences(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


def outputVar(l, voc):
    indexes_batch = [indexesFromSentences(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len


# print("input variable : ")
# print(input_variables)
# print("lengths : ", lengths)
# print("target_variable : ")
# print(target_variable)
# print("mask : ")
# print(mask)
# print("max_target_len : ", max_target_len)

print("That's the end to 217 lines of dataProcessing !")


# defining the model

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(
            0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        outputs = outputs[:, :, :self.hidden_size] + \
            outputs[:, :, self.hidden_size:]
        return outputs, hidden


class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden*encoder_ouput, dim=2)

    def forward(self, hidden, encoder_outputs):
        attn_energies = self.dot_score(hidden, encoder_outputs)
        attn_energies = attn_energies.t()
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


# now write the encoder

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden


def maskNLLLoss(decoder_out, target, mask):
    nTotal = mask.sum
    target = target.view(-1, 1)
    gathered_tensor = torch.gather(decoder_out, 1, target)
    crossEntropy = -torch.log(gathered_tensor)
    loss = crossEntropy.masked_select(mask)
    loss = loss.mean()
    loss = loss.to(device)
    return loss, nTotal.item()

# training !!


# pretrain testing
small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs)
                          for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches
# lengths = lengths.view(-1, 1)
print("input_variable shape : ", input_variable.shape)
print("lengths shape : ", lengths.shape)
print("target_variable shape : ", target_variable.shape)
print("mask_shape : ", mask.shape)
print("max_target_len : ", max_target_len)

hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
attn_model = 'dot'
embedding = nn.Embedding(voc.num_words, hidden_size)

encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(
    attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
decoder = decoder.to(device)
encoder = encoder.to(device)

encoder.train()
decoder.train()

encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.0001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.0001)
encoder_optimizer.zero_grad()
decoder_optimizer.zero_grad()

input_variable = input_variable.to(device)
lengths = lengths.to('cpu')
target_variable = target_variable.to(device)
mask = mask.to(device)

loss = 0
print_losses = []
n_totals = 0
print(type(input_variable))
encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
print("Encoder Outputs Shape :", encoder_outputs.shape)
print("Last Encoder Hidden Shape", encoder_hidden.shape)

decoder_input = torch.LongTensor(
    [[SOS_token for _ in range(small_batch_size)]])
decoder_input = decoder_input.to(device)
print("Initial Decoder Input Shape : ", decoder_input.shape)
print(decoder_input)

decoder_hidden = encoder_hidden[:decoder.n_layers]
print("Initial Decoder hidden state shape : ", decoder_hidden.shape)
print("\n")
print("-------------------------------------------------------------")
print("Now Let's look what's happening in every timestamp  of the GRU!")
print("-------------------------------------------------------------")
print("\n")
