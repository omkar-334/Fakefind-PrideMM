import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        # Input shape: (batch_size, seq_len, channels)
        x = x.transpose(1, 2).contiguous()
        # Transpose for conv: (batch_size, channels, seq_len)
        x = self.conv(x)
        # Transpose back: (batch_size, seq_len, channels)
        return x.transpose(1, 2)


class ConvAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.conv = ConvolutionalBlock(d_model, d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        conv_out = self.conv(x)
        attn_out, _ = self.attention(conv_out, conv_out, conv_out)
        return self.norm(attn_out)


class WordHierarchy(nn.Module):
    def __init__(self, d_model, num_heads=8, num_blocks=6):
        super().__init__()
        self.conv_attention_blocks = nn.ModuleList(
            [ConvAttentionBlock(d_model, num_heads) for _ in range(num_blocks)],
        )
        self.concat_projection = nn.Linear(d_model * num_blocks, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        outputs = [block(x) for block in self.conv_attention_blocks]

        output = torch.cat(outputs, dim=-1)

        projected = self.concat_projection(output)

        return self.norm(projected)


class SentenceHierarchy(nn.Module):
    def __init__(self, d_model, num_heads=8, num_blocks=5):
        super().__init__()
        self.conv_attention_blocks = nn.ModuleList(
            [ConvAttentionBlock(d_model, num_heads) for _ in range(num_blocks)],
        )

        # Conv MHSA(Tanh & ELU)
        self.conv_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        # Elementwise multiplication and normalization
        self.norm = nn.LayerNorm(d_model)

        # Conv MHSA
        self.target_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        # document projection
        self.doc_projection = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Dropout(0.1), nn.Linear(d_model, d_model))

    def forward(self, x):
        outputs = [block(x) for block in self.conv_attention_blocks]
        output = sum(outputs) / len(outputs)

        attn, _ = self.conv_attention(output, output, output)

        attn_tanh = torch.tanh(attn)
        attn_elu = F.elu(attn)

        combined_features = self.norm(attn_tanh * attn_elu)
        target_attn_output, _ = self.target_attention(combined_features, combined_features, combined_features)
        doc_embedding = self.doc_projection(target_attn_output)
        return doc_embedding


class HCAN(nn.Module):
    def __init__(self, d_model=768, num_heads=8, num_blocks=6):
        super().__init__()

        # Word hierarchy
        self.word_hierarchy = WordHierarchy(d_model, num_heads, num_blocks)

        # Sentence hierarchy
        self.sentence_hierarchy = SentenceHierarchy(d_model, num_heads, num_blocks)

    def forward(self, documents):
        if documents.dim() == 4:
            batch_size, num_sentences, num_words, embedding_dim = documents.shape
        elif documents.dim() == 3:
            batch_size, num_sentences, embedding_dim = documents.shape
            num_words = 1

            # Reshape to (batch_size, num_sentences, 1, embedding_dim)
            documents = documents.unsqueeze(2)

        reshaped_docs = documents.view(batch_size * num_sentences, num_words, embedding_dim)

        word_embeddings = self.word_hierarchy(reshaped_docs)

        sentence_embeddings = word_embeddings.mean(dim=1)
        sentence_embeddings = sentence_embeddings.view(batch_size, num_sentences, -1)

        doc_embeddings = self.sentence_hierarchy(sentence_embeddings)
        doc_embeddings = doc_embeddings.mean(dim=1)

        return doc_embeddings
