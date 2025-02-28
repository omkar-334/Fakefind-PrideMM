import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt")


def get_max_lengths(texts):
    max_sent_len = 0  # Max number of sentences in a document
    max_seq_len = 0  # Max number of words in a sentence

    for text in texts:
        sentences = sent_tokenize(text)  # Split into sentences
        max_sent_len = max(max_sent_len, len(sentences))

        for sent in sentences:
            words = word_tokenize(sent)  # Split into words
            max_seq_len = max(max_seq_len, len(words))

    return max_sent_len, max_seq_len
