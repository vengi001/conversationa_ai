import json
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from utils import tokenize, bag_of_words, stem
from torch.utils.data import DataLoader, Dataset
from model import NeuralNet
import torch.nn as nn

class ChatDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return torch.tensor(self.x[index], dtype=torch.float32), torch.tensor(self.y[index], dtype=torch.long)

with open('intents.json', 'r') as intent:
    intents = json.load(intent)

all_words = []
tags = []
xy = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        tag = intent["tag"]
        tags.append(tag)
        tokenized_pattern = tokenize(pattern)
        all_words.extend(tokenized_pattern)
        xy.append((tokenized_pattern, tag))


ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
# label_encoder = LabelEncoder()
# encoded_labels = label_encoder.fit_transform(tags)


x_train = [bag_of_words(pattern, all_words) for pattern,_ in xy]
y_train = [tags.index(tag) for _,tag in xy]

x_train = np.array(x_train)
y_train = np.array(y_train)

input_size = len(x_train[0])
hidden_size = 8
num_classes = len(tags)
batch_size = 8
learning_rate = 0.001
num_epochs = 500

dataset = ChatDataset(x_train, y_train)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = NeuralNet(input_size, hidden_size, num_classes)
criterian = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for words, labels in data_loader:
        outputs = model(words)
        loss = criterian(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": num_classes,
"all_words": all_words,
"tags": tags
}
torch.save(data, "model.pth")
print("Training complete. Model saved to model.pth")

