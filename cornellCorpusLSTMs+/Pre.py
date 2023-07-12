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

# preprocessing

lines_filepath = os.path.join("wData", "movie_lines.txt")
conv_filepath = os.path.join("wData", "movie_conversations.txt")

# with open(lines_filepath, 'rb') as file:
#     lines = file.readlines()
# for line in lines[:8]:
#     print(line.strip())

# split to lineID, characterID, movieID, character, text

line_fields = ["lineID", "characterID", "movieID", "character", "text"]
lines = {}
with open(lines_filepath, 'r', encoding="iso-8859-1") as f:
    for line in f:
        values = line.split(" +++$+++ ")
        # extract fields
        lineObj = {}
        for i, field in enumerate(line_fields):
            lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj

conv_fields = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
conversations = []
with open(conv_filepath, 'r', encoding="iso-8859-1") as f1:
    for line in f1:
        values = line.split(" +++$+++ ")
        convObj = {}
        for i, field in enumerate(conv_fields):
            convObj[field] = values[i]
        lineIds = eval(convObj["utteranceIDs"])
        convObj["lines"] = []
        for lineId in lineIds:
            convObj["lines"].append(lines[lineId])
        conversations.append(convObj)


# processing the dataset

qa_pairs = []
for conversation in conversations:
    for i in range(len(conversation["lines"]) - 1):
        inputLine = conversation["lines"][i]["text"].strip()
        targetLine = conversation["lines"][i+1]["text"].strip()

        if inputLine and targetLine:
            qa_pairs.append([inputLine, targetLine])

datafile = os.path.join("wData", "formatted_movie_lines.txt")
delimiter = '\t'
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

print("\nWriting newly formatted file . . . ")
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter)
    for pair in qa_pairs:
        writer.writerow(pair)
print("Done writing to file")

# datafile = os.path.join("wData", "formatted_movie_lines.txt")
# with open(datafile, 'rb') as file:
#     lines = file.readlines()
# for line in lines[:8]:
#     print(line)
