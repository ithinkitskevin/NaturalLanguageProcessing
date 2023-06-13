import pickle
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_and_pad(text, word2idx, pad_token='<pad>', init_token='<s>', eos_token='</s>', unk_token='<unk>'):
    tokens = text.split()
    padded_tokens = [init_token] + tokens + [eos_token]
    indices = [word2idx.get(word, word2idx[unk_token]) for word in padded_tokens]
    
    return indices

def predict_tags(model, input_text, word2idx, idx2tag):
    model.eval()
    tokenized_input = tokenize_and_pad(input_text, word2idx)
    input_tensor = torch.tensor([tokenized_input]).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
    
    predicted_indices = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
    predicted_tags = [idx2tag[idx] for idx in predicted_indices][1:-1]

    return predicted_tags

def createFile(model, textFile, outputFile, tag2idx, word2idx):
    with open(textFile, 'r') as input_file, open(outputFile, 'w') as output_file:
        indexs = []
        words = []
        tags = []
        for line in input_file:
            if not line.strip():
                if len(words) > 0 and len(tags) > 0:
                    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

                    new_text = " ".join(words)
                    predicted_tags = predict_tags(model, new_text, word2idx, idx2tag)

                    for i in range(len(indexs)):
                        index = indexs[i]
                        word = words[i]
                        tag = tags[i]
                        prediction = predicted_tags[i]

                        predictionLine = str(index) + " " + str(word) + " " + str(tag) + " " + str(prediction) + "\n"
                        output_file.write(predictionLine)
                    
                    indexs = []
                    words = []
                    tags = []
                    output_file.write("\n")
            else:
                index, word, tag = line.strip().split()
                indexs.append(index)
                words.append(word)
                tags.append(tag)

def createFileTest(model, textFile, outputFile, tag2idx, word2idx):
    with open(textFile, 'r') as input_file, open(outputFile, 'w') as output_file:
        indexs = []
        words = []
        for line in input_file:
            if not line.strip():
                if len(words) > 0:
                    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

                    new_text = " ".join(words)
                    predicted_tags = predict_tags(model, new_text, word2idx, idx2tag)

                    for i in range(len(indexs)):
                        index = indexs[i]
                        word = words[i]
                        prediction = predicted_tags[i]

                        predictionLine = str(index) + " " + str(word) + " " + str(prediction) + "\n"
                        output_file.write(predictionLine)
                    
                    indexs = []
                    words = []
                    output_file.write("\n")
            else:
                index, word = line.strip().split()
                indexs.append(index)
                words.append(word)

# Task 1

with open('word2idx_task_1.pickle', 'rb') as handle:
    word2idx = pickle.load(handle)
with open('tag2idx_task_1.pickle', 'rb') as handle:
    tag2idx = pickle.load(handle)

vocab_size = len(word2idx)
num_tags = len(tag2idx)

embedding_dim = 100
hidden_dim = 256
num_layers = 1
dropout = 0.33
linear_output_dim = 128

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, linear_output_dim, embedding_dim, hidden_dim, num_layers, dropout):
        super(BiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim * 2, linear_output_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(linear_output_dim, num_tags)

    def forward(self, x):
        # TODO lol fix name conv pls thx
        x = self.embedding(x)
        # print("NO GLOVE Forward",x)
        x, _ = self.lstm(x)
        x = self.linear1(x)
        x = self.elu(x)
        x = self.dropout(x)
        logits = self.linear2(x)

        return logits
    
loaded_model = BiLSTM(len(word2idx), linear_output_dim, embedding_dim, hidden_dim, num_layers, dropout)

saved_state_dict = torch.load("model_task_1_final.pt")
loaded_model.load_state_dict(saved_state_dict)
loaded_model.eval()

createFile(loaded_model, "data/dev", "dev1.out", tag2idx, word2idx)
createFileTest(loaded_model, "data/test", "test1.out", tag2idx, word2idx)


# Task 2

def tokenize_and_pad2(text, word2idx, pad_token='<pad>', init_token='<s>', eos_token='</s>', unk_token='<unk>'):
    tokens = text.split()

    lower_tokens = text.lower().split()
    padded_tokens = [init_token] + lower_tokens + [eos_token]
    indices = [word2idx.get(word, word2idx[unk_token]) for word in padded_tokens]
    
    upper_indices = [0] + [int(token[0].isupper()) for token in tokens] + [0]
    
    return indices, upper_indices

def predict_tags2(model, input_text, word2idx, idx2tag):
    model.eval()
    tokenized_input, upper_input = tokenize_and_pad2(input_text, word2idx)
    input_tensor = torch.tensor([tokenized_input]).to(device)
    upper_tensor = torch.tensor([upper_input]).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor, upper_tensor)
    
    predicted_indices = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
    predicted_tags = [idx2tag[idx] for idx in predicted_indices][1:-1]

    return predicted_tags

def createFile2(model, textFile, outputFile, tag2idx, word2idx):
    with open(textFile, 'r') as input_file, open(outputFile, 'w') as output_file:
        indexs = []
        words = []
        tags = []
        for line in input_file:
            if not line.strip():
                if len(words) > 0 and len(tags) > 0:
                    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

                    new_text = " ".join(words)
                    predicted_tags = predict_tags2(model, new_text, word2idx, idx2tag)

                    for i in range(len(indexs)):
                        index = indexs[i]
                        word = words[i]
                        tag = tags[i]
                        prediction = predicted_tags[i]

                        predictionLine = str(index) + " " + str(word) + " " + str(tag) + " " + str(prediction) + "\n"
                        output_file.write(predictionLine)
                    
                    indexs = []
                    words = []
                    tags = []
                    output_file.write("\n")
            else:
                index, word, tag = line.strip().split()
                indexs.append(index)
                words.append(word)
                tags.append(tag)

def createFileTest2(model, textFile, outputFile, tag2idx, word2idx):
    with open(textFile, 'r') as input_file, open(outputFile, 'w') as output_file:
        indexs = []
        words = []
        for line in input_file:
            if not line.strip():
                if len(words) > 0:
                    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

                    new_text = " ".join(words)
                    predicted_tags = predict_tags2(model, new_text, word2idx, idx2tag)

                    for i in range(len(indexs)):
                        index = indexs[i]
                        word = words[i]
                        prediction = predicted_tags[i]

                        predictionLine = str(index) + " " + str(word) + " " + str(prediction) + "\n"
                        output_file.write(predictionLine)
                    
                    indexs = []
                    words = []
                    output_file.write("\n")
            else:
                index, word = line.strip().split()
                indexs.append(index)
                words.append(word)

with open('word2idx_task_2.pickle', 'rb') as handle:
    word2idx = pickle.load(handle)
with open('tag2idx_task_2.pickle', 'rb') as handle:
    tag2idx = pickle.load(handle)

vocab_size = len(word2idx)
num_tags = len(tag2idx)

embedding_dim = 100
hidden_dim = 256
num_layers = 1
dropout = 0.33
linear_output_dim = 128

class BiLSTM2(nn.Module):
    def __init__(self, vocab_size, linear_output_dim, embedding_dim, hidden_dim, num_layers, dropout):
        super(BiLSTM2, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.upper_embedding = nn.Embedding(2, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim * 2, linear_output_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(linear_output_dim, num_tags)

    def forward(self, x, upper_x):
        x = self.embedding(x)
        upper_x = self.upper_embedding(upper_x)
        x = torch.cat([x, upper_x], dim=-1)
        x, _ = self.lstm(x)
        x = self.linear1(x)
        x = self.elu(x)
        x = self.dropout(x)
        logits = self.linear2(x)

        return logits

loaded_model = BiLSTM2(len(word2idx), linear_output_dim, embedding_dim, hidden_dim, num_layers, dropout)

saved_state_dict = torch.load("model_2.pt")
loaded_model.load_state_dict(saved_state_dict)
loaded_model.eval()

createFile2(loaded_model, "data/dev", "dev2.out", tag2idx, word2idx)
createFileTest2(loaded_model, "data/test", "test2.out", tag2idx, word2idx)

# Extra Credit

def tokenize_and_pad3(text, word2idx, char2idx, pad_token='<pad>', init_token='<s>', eos_token='</s>', unk_token='<unk>'):
    tokens = text.split()

    lower_tokens = text.lower().split()
    padded_tokens = [init_token] + lower_tokens + [eos_token]
    indices = [word2idx.get(word, word2idx[unk_token]) for word in padded_tokens]
    
    upper_indices = [0] + [int(token[0].isupper()) for token in tokens] + [0]

    char_indices = [[char2idx.get(char, char2idx[unk_token]) for char in word] for word in tokens]
    max_word_len = max([len(word_chars) for word_chars in char_indices]) + 2
    char_indices = [[char2idx[pad_token]] * max_word_len] + char_indices + [[char2idx[pad_token]] * max_word_len]
    char_indices_padded = [word_chars + [char2idx[pad_token]] * (max_word_len - len(word_chars)) for word_chars in char_indices]

    return indices, upper_indices, char_indices_padded


def predict_tags3(model, input_text, word2idx, char2idx, idx2tag):
    model.eval()
    tokenized_input, upper_input, char_input = tokenize_and_pad3(input_text, word2idx, char2idx)
    input_tensor = torch.tensor([tokenized_input]).to(device)
    upper_tensor = torch.tensor([upper_input]).to(device)
    char_input_tensor = torch.tensor([char_input]).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor, upper_tensor, char_input_tensor)
    
    predicted_indices = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
    predicted_tags = [idx2tag[idx] for idx in predicted_indices][1:-1]

    return predicted_tags

def createFile3(model, textFile, outputFile, tag2idx, word2idx):
    with open(textFile, 'r') as input_file, open(outputFile, 'w') as output_file:
        indexs = []
        words = []
        tags = []
        for line in input_file:
            if not line.strip():
                if len(words) > 0 and len(tags) > 0:
                    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

                    new_text = " ".join(words)
                    predicted_tags = predict_tags3(model, new_text, word2idx, char2idx, idx2tag)

                    for i in range(len(indexs)):
                        index = indexs[i]
                        word = words[i]
                        tag = tags[i]
                        prediction = predicted_tags[i]

                        predictionLine = str(index) + " " + str(word) + " " + str(tag) + " " + str(prediction) + "\n"
                        output_file.write(predictionLine)
                    
                    indexs = []
                    words = []
                    tags = []
                    output_file.write("\n")
            else:
                index, word, tag = line.strip().split()
                indexs.append(index)
                words.append(word)
                tags.append(tag)

def createFileTest3(model, textFile, outputFile, tag2idx, word2idx):
    with open(textFile, 'r') as input_file, open(outputFile, 'w') as output_file:
        indexs = []
        words = []
        for line in input_file:
            if not line.strip():
                if len(words) > 0:
                    idx2tag = {idx: tag for tag, idx in tag2idx.items()}

                    new_text = " ".join(words)
                    predicted_tags = predict_tags3(model, new_text, word2idx, char2idx, idx2tag)

                    for i in range(len(indexs)):
                        index = indexs[i]
                        word = words[i]
                        prediction = predicted_tags[i]

                        predictionLine = str(index) + " " + str(word) + " " + str(prediction) + "\n"
                        output_file.write(predictionLine)
                    
                    indexs = []
                    words = []
                    output_file.write("\n")
            else:
                index, word = line.strip().split()
                indexs.append(index)
                words.append(word)

char_embedding_dim = 30
embedding_dim = 100
hidden_dim = 256
num_layers = 1
dropout = 0.33
linear_output_dim = 128

class BiLSTM_CNN(nn.Module):
    def __init__(self, vocab_size, char_vocab_size, num_tags, char_embedding_dim, embedding_dim, hidden_dim, num_layers, dropout, linear_output_dim):
        super(BiLSTM_CNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.upper_embedding = nn.Embedding(2, embedding_dim)
        
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)
        self.char_cnn = nn.Conv1d(char_embedding_dim, embedding_dim, kernel_size=3)
        
        self.lstm = nn.LSTM(embedding_dim * 3, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim * 2, linear_output_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(linear_output_dim, num_tags)

    def forward(self, x, upper_x, chars):
        x = self.embedding(x)
        upper_x = self.upper_embedding(upper_x)
        
        chars = self.char_embedding(chars)
        batch_size, max_seq_len, max_word_len, _ = chars.shape
        chars = chars.view(batch_size * max_seq_len, max_word_len, -1).permute(0, 2, 1)
        # print(chars)

        char_features = self.char_cnn(chars)
        char_features = nn.functional.relu(char_features)
        char_features, _ = torch.max(char_features, dim=-1)
        char_features = char_features.view(batch_size, max_seq_len, -1)
        # print(char_features)
        
        x = torch.cat([x, upper_x, char_features], dim=-1)
        x, _ = self.lstm(x)
        x = self.linear1(x)
        x = self.elu(x)
        x = self.dropout(x)
        logits = self.linear2(x)

        return logits
    
with open('char2idx_task_3.pickle', 'rb') as handle:
    char2idx = pickle.load(handle)
with open('word2idx_task_3.pickle', 'rb') as handle:
    word2idx = pickle.load(handle)
with open('tag2idx_task_3.pickle', 'rb') as handle:
    tag2idx = pickle.load(handle)

vocab_size = len(word2idx)
char_vocab_size = len(char2idx)
num_tags = len(tag2idx)

loaded_model = BiLSTM_CNN(vocab_size, char_vocab_size, num_tags, char_embedding_dim, embedding_dim, hidden_dim, num_layers, dropout, linear_output_dim)

saved_state_dict = torch.load("model_3.pt")
loaded_model.load_state_dict(saved_state_dict)
loaded_model.eval()

createFile3(loaded_model, "data/dev", "dev3.out", tag2idx, word2idx)
createFileTest3(loaded_model, "data/test", "test3.out", tag2idx, word2idx)