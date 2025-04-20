import os

import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

# https://github.com/werywjw/MultiClimate/blob/main/models/finetune_BERT%2BViT.py


def make_df(dataset_root="/workspace/MultiClimate/dataset"):
    test_videos = ["ACCFP", "CCAH", "CCSAD", "CCUIM", "EIB", "EWCC", "GGCC", "SCCC", "TICC", "WICC"]
    val_videos = ["CCGFS", "CCIAP", "CICC", "EFCC", "FIJI", "HCCAB", "HRDCC", "HUSNS", "MACC", "SAPFS"]
    train_videos = [
        "ACCC",
        "AIAQ",
        "AIDT",
        "AMCC",
        "BDCC",
        "BECCC",
        "BWFF",
        "CBAQC",
        "CCBN",
        "CCBNN",
        "CCCBL",
        "CCCP",
        "CCCS",
        "CCD",
        "CCFS",
        "CCFWW",
        "CCH",
        "CCHES",
        "CCIAA",
        "CCIAH",
        "CCICD",
        "CCIS",
        "CCISL",
        "CCMA",
        "CCSC",
        "CCTA",
        "CCTP",
        "CCWC",
        "CCWQ",
        "CESS",
        "COP",
        "CPCC",
        "CTCM",
        "DACC",
        "DFCC",
        "DPIC",
        "DTECC",
        "ECCDS",
        "FCC",
        "FLW",
        "FTACC",
        "HCCAE",
        "HCCAW",
        "HCCIG",
        "HCI",
        "HDWC",
        "HHVBD",
        "HSHWA",
        "HSPW",
        "IMRF",
        "INCAS",
        "MICC",
        "NASA",
        "OCCC",
        "PCOCC",
        "PWCCA",
        "RAGG",
        "RASCC",
        "RCCCS",
        "RCCS",
        "RHTCC",
        "RPDCC",
        "SDDA",
        "SLCCA",
        "SSTCC",
        "TCBCC",
        "TECCC",
        "TIOCC",
        "TIYH",
        "TTFCC",
        "TUCC",
        "UKCC",
        "VFVCC",
        "VPCC",
        "WCCA",
        "WFHSW",
        "WICCE",
        "WISE",
        "WTCC",
        "YPTL",
    ]
    splits = {"train": train_videos, "test": test_videos, "val": val_videos}
    alldata = []

    for split, videos in splits.items():
        for video in videos:
            csv_path = os.path.join(dataset_root, video, f"{video}.csv")

            if not os.path.exists(csv_path):
                continue

            data = pd.read_csv(csv_path, header=None, skiprows=1, names=["label", "text"])
            image_folder = os.path.join(dataset_root, video, f"{video}_frames")

            for index, row in data.iterrows():
                label = row["label"]
                transcript = row["text"]
                image_path = os.path.join(image_folder, f"{video}-{index + 1:03d}.jpg")

                if os.path.exists(image_path):
                    alldata.append((transcript, image_path, label, split))

    df = pd.DataFrame(alldata, columns=["text", "path", "label", "split"])

    df = df[["label", "path", "text", "split"]]

    df.to_csv("dataset.csv", index=False)
    return df


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)

transform = transforms.Compose(
    [
        transforms.Resize((299, 299)),  # Resize for Inception-ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
    ]
)


def process_dataset(file_name, batch_size=32):
    """Precompute text and image embeddings and save them in a pickle file."""
    df = pd.read_csv(file_name)
    text_embeddings = []
    image_embeddings = []

    texts = df["text"].fillna("null").tolist()

    for i in tqdm(range(0, len(texts), batch_size), total=len(texts) // batch_size, desc="Processing text embeddings"):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = extract_bert_embeddings(batch_texts)
        text_embeddings.extend(batch_embeddings.cpu().numpy())

    for image_path in tqdm(df["path"], total=len(df), desc="Processing images"):
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        image_embeddings.append(image_tensor.cpu().numpy())

    df["text_embedding"] = text_embeddings
    df["image_embedding"] = image_embeddings

    df.to_pickle("features.pkl")
    return df


def extract_bert_embeddings(texts):
    """Extract BERT embeddings efficiently using GPU."""
    encoding = tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding="max_length")

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)

    return outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token embeddings


# main_text = """LGBTQ stands for Lesbian, Gay, Bisexual, Transgender, and Queer, representing diverse sexual orientations and gender identities advocating for equality and acceptance. LGBTQ individuals face unique challenges, including discrimination and stigma, while also fostering vibrant communities, pride movements, and legal rights advancements worldwide, promoting inclusivity and representation."""
main_text = """Climate change refers to long-term shifts in temperatures and weather patterns. Such shifts can be natural, due to changes in the sun's activity or large volcanic eruptions. But since the 1800s, human activities have been the main driver of climate change, primarily due to the burning of fossil fuels like coal, oil and gas."""
main_text = extract_bert_embeddings(main_text)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.set_default_dtype(torch.float32)


class Dataset(Dataset):
    def __init__(self, pkl, label, split, image_size=224):
        super(Dataset, self).__init__()
        self.label = label

        self.image_size = image_size

        self.df = pd.read_pickle(pkl)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

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
