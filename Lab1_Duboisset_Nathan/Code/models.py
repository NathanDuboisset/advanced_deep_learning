
import torch
import torch.nn as nn

class DeepSets(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(DeepSets, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        
        ############## Task 3
        
        embedded = self.embedding(x)
        
        pre_activation = self.fc1(embedded)
        activated_elements = self.tanh(pre_activation)
        
        aggregated = torch.sum(activated_elements, dim=1)
        
        final_output = self.fc2(aggregated)
        
        return final_output.squeeze()


class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        
        ############## Task 4
        
        embedded = self.embedding(x)
        
        lstm_outputs, _ = self.lstm(embedded)
        
        last_hidden_state = lstm_outputs[:, -1, :]
        
        final_output = self.fc(last_hidden_state)
        
        return final_output.squeeze()
