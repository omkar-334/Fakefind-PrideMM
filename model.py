import torch
import torch.nn as nn

from hcan import HCAN
from inception_resnet_v2 import Inception_ResNetv2
from knowledge_extractor import KnowledgeExtractor


class MLP(nn.Module):
    def __init__(self, input_dim=450, hidden_dim=450, num_classes=3):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Added dropout layer

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout
        x = self.output_layer(x)
        return x


class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.5):
        super(BiGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        if x.dim() == 2:  # If x is (seq_length, input_size), add batch dim
            x = x.unsqueeze(0)

        # x shape: (batch_size, seq_length, input_size)
        h0 = torch.zeros(
            self.num_layers * 2,
            x.size(0),
            self.hidden_size,
        ).to(x.device)

        out, _ = self.gru(x, h0)

        # out shape: (batch_size, seq_length, hidden_size * 2)
        out = self.fc(out[:, -1, :])
        return out


class Final(nn.Module):
    def __init__(self):
        super(Final, self).__init__()

        self.hcan = HCAN()
        self.bigru1 = BiGRU(input_size=768, hidden_size=100, output_size=100, num_layers=2)
        self.bigru2 = BiGRU(input_size=100, hidden_size=100, output_size=100, num_layers=2)

        self.i_resnet = Inception_ResNetv2()
        self.KE = KnowledgeExtractor()
        self.MLP = MLP()

        self.text_attention_fc = nn.Linear(3, 450)
        self.image_attention_fc = nn.Linear(1000, 450)

    def forward(self, text, reply, image):
        text_feat = self.bigru1(self.hcan(text))
        reply_feat = self.bigru1(self.hcan(reply))

        combined_text_feat = self.bigru2(text_feat + reply_feat)

        extracted_knowledge = self.KE(combined_text_feat)

        image_feat = self.i_resnet(image)

        # Apply attention mechanism to text and image features
        text_attention = torch.tanh(self.text_attention_fc(extracted_knowledge))
        image_attention = torch.tanh(self.image_attention_fc(image_feat))

        final_feature = text_attention + image_attention
        output = self.MLP(final_feature)
        return output
