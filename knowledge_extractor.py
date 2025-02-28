import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

# This KE block has two layers, the TS-O encoder and OSN.
# 1) TS-O Encoder: As an example, c ∈ PRTS contains
# the TSC representation of post “P” and replies “R” with
# “n” tokens (r1, r2, . . . , rn) and an opinion instance “O”
# with “m” tokens (o1, o2, . . . , om). To an opinion “O” that
# is too related to a stance label “y.” TS-opinion (TS-O)-
# encoder (TS-Oe) uses bidirectional encoder representations
# from transformers (BERT) to encode text-social representations
# “R” and opinion “O” concurrently. This work proposes
# processing R and O along with a single token sequence
# ([CLS], r1, r2, . . . , rn, [SEP], o1, o2, . . . , om, [SEP]). Later,
# TS-Oe builds input embedding by also transforming that into
# the combined embedding like BERT. Accurately, an input
# embedding being denoted like X = (x1, x2, . . . , xl ), wherein
# xi ∈ Rk to be the embedding through summating on the token,
# location embeddings, and segment, and “l” is the determined
# dimension of a token order. Later, creating the input set X,
# TS-OE changes that into the joint text and social with opinion
# embedding. At the jth layer, a text-target set is represented
# as H( j) = (h1, h2, . . . , hl ), wherein hi ∈ Rk has similar
# dimensions by an equivalent input xi , and H( j) is reached
# as in the following equations:
# Y = H(J−1) (3)
# Multi_Head(Y ) = Concat(H1, . . . , Hm)WO (5)
# Z = LN(Y + Multi_Head (Y )) (6)
# H( j) = LN(Z + MLP(Z)) (7)
# wherein WQi ,WKi ,WVi
# ∈ Rd×(d/m), Wo ∈ Rd×d are learnable
# variables, LN is layer normalization, and H(0) = X. This
# complete model stacks “J ” such layers and the end layer H(J ).
# This article has been selected as our final TSC opinion
# embedding H′.


class KnowledgeExtractor(nn.Module):
    def __init__(self, bert_model="bert-base-uncased", hidden_size=768, num_classes=3, num_filters=100, filter_sizes=[2, 3, 4], dropout_prob=0.5):
        """
        Complete Knowledge Extractor with TS-O Encoder and Opinion Convolution Network

        Args:
            bert_model: Pre-trained BERT model name
            hidden_size: Hidden dimension size
            num_classes: Number of stance classes
            num_filters: Number of convolutional filters per stance
            filter_sizes: List of filter sizes for convolutions
            dropout_prob: Dropout probability
        """
        super(KnowledgeExtractor, self).__init__()

        # 1. TS-O Encoder
        self.ts_o_encoder = nn.Module()
        self.ts_o_encoder.bert = transformers.BertModel.from_pretrained(bert_model)
        self.ts_o_encoder.dropout = nn.Dropout(dropout_prob)

        # 2. Opinion Convolution Network
        self.ocn = OpinionConvolutionNetwork(
            embedding_dim=hidden_size,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            num_classes=num_classes,
            dropout_prob=dropout_prob,
        )

    def forward(self, input_ids, token_type_ids=None):
        """
        Forward pass through the complete Knowledge Extractor

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]

        Returns:
            stance_probs: Predicted stance probabilities [batch_size, num_classes]
        """
        # 1. Process through TS-O Encoder
        outputs = self.ts_o_encoder.bert(
            input_ids=input_ids.to(torch.long),
            attention_mask=torch.ones_like(input_ids).to(torch.long),
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        sequence_output = self.ts_o_encoder.dropout(sequence_output)

        # 2. Process through Opinion Convolution Network
        stance_probs = self.ocn(sequence_output)

        return stance_probs


# 2) Opinion Convolution Network: Once the joint TS-O
# embedding is generated from the TS-O encoder, the OCN
# layer is applied to extract stance-indicative features. This
# layer is used to remove structural information from the input
# sequence. In addition, three CNNs are deployed: neutral
# opinion convolution (NOC), favor opinion convolution (FOC),
# and against opinion convolution (AOC). The corresponding
# stance-indicative features are predicted to be extracted from
# each OCN. Every OCN uses the combined TSC opinion
# embedding from the TS-O encoder layer as its input. Let
# hi ∈ Rk represent a “k” dimensional output embedding
# corresponding to the ith token within the TS-O encoder’s input
# sequence. The equation for the line of length “l” to be
# H = h1 ⊕ h2 ⊕ · · · ⊕ hl (8)
# wherein H ∈ Rl×k and ⊕ are the concatenation operators.
# This work uses different filters to convolution it after that.
# A convolution operation uses a filter w ∈ Rh×k to create a new
# feature by applying it to the “h” tokens window. A feature
# ci ∈ R, for instance, is produced from a window of tokens
# hi:i+h−1 by the following equation:
# ci = f (w ⊙ hi:i+h−1 + b) (9)
# wherein the function “ f ” is nonlinear. For producing a feature
# map, this filter is functional to every potential window of
# tokens within the sequence in the following equation:
# c = (c1, c2, . . . , cn−h+1). (10)
# This work later employs a max-over-time pooling operation
# upon a feature map by also taking the maximum value
# ˆ c = max{c} to be the feature corresponding to such a specific
# filter. Lastly, the generation of elements from all filters gets
# cohesive to form a single high-level feature vector.
# As a result, to find a feature vector using the expression
# ˆ h ∈ Rp, where “p” is the number of filters for each OCN.
# The following equation criteria are used to define the stance
# scores:
# α0 = W0
# ˆ h0 + b0
# α1 = W1
# ˆ h1 + b1
# ...
# αn = Wn
# ˆ hn + bn (11)
# wherein W ∈ R1×d and b ∈ R label a stance-specific
# linear transformation. The higher the value of αi , the most
# probably “i” to be the valid stance label.
# Let us assume α0, α1, . . . , αn to signify an opinion scores
# vector. This confidence scores “α” to be later normalized for
# probabilities applying the SoftMax operation in the following
# equation:
# α = [α0, α1, . . . , αn]
# ˆy = SoftMax(α) (12)
# wherein ˆy ∈ Rn seems to be the vector of forecast probability
# to the “n” stances.
# As a result of the global average pooling’s (GAP) large
# pooling size, which is symbolized by the 7 × 7 and
# 8 × 8 layouts, it also significantly affects the transfer of
# the gradient. For example, consider the network’s final loss
# X(Ya, Y b), where (Ya, Y b) denotes the actual and predicted
# labels, respectively. The SoftMax function decides the
# Y p value in the following equation:
# Y p = f
# 􀀀
# PxY c + Qs
# 
# . (13)
# In (20), f () stands for the SoftMax function, and Px and
# Qy represent the SoftMax layer weights. When the SoftMax
# layer’s gradient is multiplied, this work obtains the SoftMax
# layer’s gradient
# ∂ Z
# 􀀀
# Ya, Y b
# ∂Y c
# =
# ∂ Z
# 􀀀
# Ya, Y b
# ∂Y b
# +
# ∂Ya
# ∂ f
# +
# ∂ f
# ∂Y c . (14)
# Like the often-used 77 pooling region, the GAP is typically
# huge. The losses transmitted from the classification layer are
# averaged by 49 to ensure that the backpropagation keeps the
# total gradients before and after pooling.


class OpinionConvolutionNetwork(nn.Module):
    """
    This network consists of three specific convolution networks:
    - Neutral Opinion Convolution (NOC)
    - Favor Opinion Convolution (FOC)
    - Against Opinion Convolution (AOC)
    """

    def __init__(self, embedding_dim=768, num_filters=100, filter_sizes=[2, 3, 4], num_classes=3, dropout_prob=0.5):
        """
        Args:
            embedding_dim: Dimension of input embeddings from TS-O encoder
            num_filters: Number of convolutional filters per stance
            filter_sizes: List of filter sizes (h in the paper)
            num_classes: Number of stance classes (typically 3: neutral, favor, against)
            dropout_prob: Dropout probability
        """
        super(OpinionConvolutionNetwork, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes

        # Create stance-specific convolution networks (NOC, FOC, AOC)
        self.stance_convs = nn.ModuleList()

        # For each stance (neutral, favor, against), create a set of convolutions with different filter sizes
        for _ in range(num_classes):
            stance_specific_convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
            self.stance_convs.append(stance_specific_convs)

        self.dropout = nn.Dropout(dropout_prob)
        self.stance_linear = nn.ModuleList([nn.Linear(len(filter_sizes) * num_filters, 1) for _ in range(num_classes)])

    def forward(self, x):
        """
        Args:
            x: Input tensor from TS-O encoder [batch_size, seq_len, embedding_dim]

        Returns:
            stance_probs: Predicted stance probabilities [batch_size, num_classes]
        """

        # Add channel dimension for Conv2d [batch_size, 1, seq_len, embedding_dim]
        x = x.unsqueeze(1)

        stance_features = []

        for i in range(self.num_classes):
            conv_outputs = []

            for conv in self.stance_convs[i]:
                # [batch_size, num_filters, seq_len-filter_size+1, 1]
                conv_out = F.relu(conv(x)).squeeze(3)

                # [batch_size, num_filters, 1]
                pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)

                conv_outputs.append(pooled)

            # [batch_size, num_filters * len(filter_sizes)]
            stance_feature = torch.cat(conv_outputs, dim=1)
            stance_feature = self.dropout(stance_feature)

            # Apply stance-specific linear transformation (equation 11)
            # [batch_size, 1]
            stance_score = self.stance_linear[i](stance_feature)
            stance_features.append(stance_score)

        # Concatenate stance scores from all stance-specific networks
        # [batch_size, num_classes]
        stance_scores = torch.cat(stance_features, dim=1)

        stance_probs = F.softmax(stance_scores, dim=1)
        return stance_probs
