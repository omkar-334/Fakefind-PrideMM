import os

import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")


def process_dataset(cfg, batch_size=32):
    """Precompute text and image embeddings and save them in a pickle file."""
    df = pd.read_csv(cfg.info_file)
    text_embeddings = []
    image_embeddings = []

    texts = df["text"].fillna("null").tolist()

    # Process text embeddings in batches
    for i in tqdm(range(0, len(texts), batch_size), total=len(texts) // batch_size, desc="Processing text embeddings"):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = extract_bert_embeddings(batch_texts)
        text_embeddings.extend(batch_embeddings)

    # Process images
    for image_fn in tqdm(df["name"], total=len(df), desc="Processing images"):
        image_path = os.path.join(cfg.img_folder, image_fn)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).numpy()
        image_embeddings.append(image_tensor)

    df["text_embedding"] = text_embeddings
    df["image_embedding"] = image_embeddings

    # Save to pickle file
    df.to_pickle("features.pkl")

    return df


def extract_bert_embeddings(text):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,  # 🔹
        padding="max_length",
    )

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)

    embeddings = outputs.last_hidden_state
    return embeddings


main_text = """LGBTQ stands for Lesbian, Gay, Bisexual, Transgender, and Queer, representing diverse sexual orientations and gender identities advocating for equality and acceptance. LGBTQ individuals face unique challenges, including discrimination and stigma, while also fostering vibrant communities, pride movements, and legal rights advancements worldwide, promoting inclusivity and representation."""
main_text = extract_bert_embeddings(main_text)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.set_default_dtype(torch.float32)

transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),  # Resize for Inception-ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
    ]
)


class Dataset(Dataset):
    def __init__(self, cfg, dataset, label, split="train", image_size=224):
        super(Dataset, self).__init__()
        self.cfg = cfg
        self.dataset = dataset
        self.split = split
        self.label = label

        self.image_size = image_size

        self.df = pd.read_pickle(cfg.pickle_file)
        self.df = self.df[self.df["split"] == self.split].reset_index(drop=True)

        # float_cols = self.df.select_dtypes(float).columns
        # self.df[float_cols] = self.df[float_cols].fillna(-1).astype("Int64")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # text = extract_bert_embeddings(text)

        # image_fn = row["name"]
        # image = Image.open(f"{self.cfg.img_folder}/{image_fn}").convert("RGB")
        # image = transform(image).unsqueeze(0)
        text = torch.tensor(row["text_embedding"]).squeeze(0)
        image = torch.tensor(row["image_embedding"]).squeeze(0)

        return (main_text, text, image, row[self.label])

    # {
    #         "image": image,
    #         "text": text,
    #         "label": row[self.label],
    #         "idx_meme": row["name"],
    #     }


def collate_fn(batch):
    main_texts, texts, images, labels = zip(*batch)

    main_texts = pad_sequence(main_texts, batch_first=True, padding_value=0)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)

    images = torch.stack(images)
    labels = torch.tensor(labels)
    return main_texts, texts_padded, images, labels


def load_dataset(cfg, split):
    dataset = Dataset(cfg=cfg, split=split, label=cfg.label)

    return dataset
