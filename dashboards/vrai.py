import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
from glob import glob

# import cv2
import pandas as pd
import seaborn as sns
import torch
import lightning as L
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.nn import functional as F
import random
from PIL import Image

SIZE = 224

with st.sidebar:
    debug_datasets_pkl = st.checkbox("Debug datasets_pkl")
if debug_datasets_pkl:
    fname = "data/VRAI/datasets_pkl/coco_imgs.pkl"
    with open(fname, "rb") as f:
        data = pickle.load(f)

    st.write(len(data))
    st.write(len(data[0]))

images_train_path = "data/VRAI/images_train/"
images_val_path = "data/VRAI/images_dev/"
images_test_path = "data/VRAI/images_test/"
images_train = glob(images_train_path + "*")
images_val = glob(images_val_path + "*")
images_test = glob(images_test_path + "*")

st.write(f"Train images: {len(images_train)}")
st.write(f"Val images: {len(images_val)}")
st.write(f"Test images: {len(images_test)}")


samples_to_show = 20


df_train = pd.DataFrame({"fname": images_train})
df_train["id"] = df_train.fname.apply(lambda x: x.split("/")[-1].split("_")[0])
df_train["camera"] = df_train.fname.apply(lambda x: x.split("/")[-1].split("_")[1])
df_train["sample"] = df_train.fname.apply(lambda x: x.split("/")[-1].split("_")[1])


df_val = pd.DataFrame({"fname": images_val})
df_val["id"] = df_val.fname.apply(lambda x: x.split("/")[-1].split("_")[0])
df_val["camera"] = df_val.fname.apply(lambda x: x.split("/")[-1].split("_")[1])
df_val["sample"] = df_val.fname.apply(lambda x: x.split("/")[-1].split("_")[1])


df_test = pd.DataFrame({"fname": images_test})
df_test["id"] = df_test.fname.apply(lambda x: x.split("/")[-1].split("_")[0])
df_test["camera"] = df_test.fname.apply(lambda x: x.split("/")[-1].split("_")[1])
df_test["sample"] = df_test.fname.apply(lambda x: x.split("/")[-1].split("_")[1])


with st.sidebar:
    show_df = st.checkbox("Show df")
if show_df:
    st.write(df_train)

with st.sidebar:
    show_sample_counts = st.checkbox("Show sample counts")
if show_sample_counts:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))
    sns.histplot(df_train.id.value_counts(), ax=ax0)
    ax0.set_xlabel("Samples count per id")
    ax0.set_ylabel("Popularity\ncount of ids with the same sample count")

    sns.histplot(df_train.id.value_counts(), ax=ax1)
    ax1.set_yscale("log")
    ax1.set_xlabel("Samples count per id")
    ax1.set_ylabel("Popularity\ncount of ids with the same sample count")
    st.pyplot(fig)

    st.write(f"Minimum samples: {min(df_train.id.value_counts())}")
    st.write(f"Maximum samples: {max(df_train.id.value_counts())}")

with st.sidebar:
    show_some_samples = st.checkbox("Show some samples")
if show_some_samples:
    st.write(sorted(images_train)[:samples_to_show])
    for fname in sorted(images_train)[:samples_to_show]:
        st.text(fname)
        img = np.array(Image.open(fname))
        st.image(img)


with st.sidebar:
    show_splits = st.checkbox("Show train/val/test")
if show_splits:
    _id = "00008382"
    cols = st.columns(3)
    with cols[0]:
        st.header("Train")
        for fname in sorted(df_train[df_train.id == _id].fname)[:samples_to_show]:
            st.text(fname)
            img = np.array(Image.open(fname))
            st.image(img)

    with cols[1]:
        st.header("Val")
        for fname in sorted(df_val[df_val.id == _id].fname)[:samples_to_show]:
            st.text(fname)
            img = np.array(Image.open(fname))
            st.image(img)

    with cols[2]:
        st.header("Test")
        for fname in sorted(df_test[df_test.id == _id].fname)[:samples_to_show]:
            st.text(fname)
            img = np.array(Image.open(fname))
            st.image(img)

vc = df_train.id.value_counts()
id_with_max_samples = list(dict(vc[np.argmax(vc) : np.argmax(vc) + 1]).keys())[0]

with st.sidebar:
    show_samples_with_max_count = st.checkbox("Show samples with maximum count")
if show_samples_with_max_count:
    max_samples_df = df_train[df_train.id == id_with_max_samples]

    fig = plt.figure()
    max_samples_df.camera.value_counts().plot.barh()
    st.pyplot(fig)

    for fname in sorted(max_samples_df.fname):
        st.text(fname)
        img = np.array(Image.open(fname))

        st.image(img)


class VRAIDataset(torch.utils.data.Dataset):
    def __init__(self, split, root_dir="data/VRAI", transform=None):
        self.split = split
        self.root_dir = root_dir
        self.transform = transform

        assert type(split) == str
        assert split in ["train", "val", "test"]

        if split == "val":
            images_path = root_dir + "/images_dev/"
        else:
            images_path = root_dir + "/images_" + split + "/"
        images = glob(images_path + "*")

        self.df = pd.DataFrame({"fname": images})

        self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.df["id"] = self.df.fname.apply(lambda x: x.split("/")[-1].split("_")[0])
        self.df["camera"] = self.df.fname.apply(
            lambda x: x.split("/")[-1].split("_")[1]
        )
        self.df["sample"] = self.df.fname.apply(
            lambda x: x.split("/")[-1].split("_")[1]
        )
        self.ids = sorted(self.df.id.unique())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        id = self.ids[idx]
        # st.write(self.df[self.df.id == id])
        # st.write(self.df[self.df.id == id].index.tolist())
        samples_with_id = self.df[self.df.id == id].index.tolist()
        # print(f"split {self.split} samples_with_id: {len(samples_with_id)}")
        if len(samples_with_id) > 1:
            row0, row1 = random.sample(samples_with_id, k=2)
        else:
            row0 = samples_with_id[0]
            row1 = row0
        row0 = self.df.loc[row0]
        row1 = self.df.loc[row1]
        fname0 = row0.fname
        fname1 = row1.fname
        camera0 = row0.camera
        camera1 = row1.camera

        img0 = np.array(Image.open(fname0))
        img1 = np.array(Image.open(fname1))

        if self.transform:
            img0 = self.transform(image=img0)["image"]
            img1 = self.transform(image=img1)["image"]
        sample = dict(
            image0=img0,
            image1=img1,
            fname0=fname0,
            fname1=fname1,
            id=id,
            camera0=camera0,
            camera1=camera1,
        )

        return sample


class ReIDModel(L.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.size = 224
        st.write(self.backbone)
        self.model = timm.create_model(self.backbone, pretrained=True, num_classes=0)

        self.features_size = self.model(torch.zeros(1, 3, self.size, self.size)).shape
        st.write(self.features_size)

        for name, param in self.model.named_parameters():
            if "stages.0" in name or "stages.1" in name:
                param.requires_grad = False
            # st.write(f"requires_grad: {param.requires_grad}, {name}")

        init_logit_scale = np.log(1 / 0.07)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * init_logit_scale)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, batch_idx, split):
        x0 = batch["image0"]
        x1 = batch["image1"]

        y0 = self.model(x0)
        y1 = self.model(x1)

        labels = torch.arange(len(y0), device=y0.device, dtype=torch.long)

        logits0 = self.logit_scale * y0 @ y1.T
        logits1 = self.logit_scale * y1 @ y0.T

        loss = (F.cross_entropy(logits0, labels) + F.cross_entropy(logits1, labels)) / 2
        self.log(f"{split}/loss", loss)
        if split == "train":
            self.log("logit_scale", self.logit_scale)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, split="train")

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, split="val")
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


if st.checkbox("Train"):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = A.Compose(
        [
            A.Resize(SIZE, SIZE),
            A.ShiftScaleRotate(
                shift_limit=0.5,
                scale_limit=(-0.5, 0.2),
                rotate_limit=90,
                p=0.9,
                border_mode=1,
            ),
            A.RGBShift(r_shift_limit=50, g_shift_limit=50, b_shift_limit=50, p=0.95),
            A.RandomBrightnessContrast(
                brightness_limit=0.6, contrast_limit=0.6, p=0.95
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    test_transform = A.Compose(
        [
            A.Resize(SIZE, SIZE),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    inv_normalize = A.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1.0 / s for s in std],
        always_apply=True,
        max_pixel_value=1.0,
    )

    ds_train = VRAIDataset(split="train", transform=train_transform)
    ds_val = VRAIDataset(split="val", transform=train_transform)
    # for i, sample in enumerate(ds):
    #     st.write(sample["image0"].shape)
    #     st.write(sample["image1"].shape)
    #     img0 = inv_normalize(image=sample["image0"].permute(1, 2, 0).numpy())["image"]
    #     img1 = inv_normalize(image=sample["image1"].permute(1, 2, 0).numpy())["image"]
    #     st.image(img0)
    #     st.image(img1)
    #     break

    backbone = "mobilenetv3_small_050.lamb_in1k"
    max_epochs = 1000

    model = ReIDModel(backbone=backbone)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=64, num_workers=8)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=64, num_workers=8)

    logger = TensorBoardLogger("tb_logs", name=backbone)

    checkpoint_callback = ModelCheckpoint(
        filename="epoch-{epoch}-val_loss-{val/loss:.6f}",
        auto_insert_metric_name=False,
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        every_n_epochs=1,
    )

    trainer = L.Trainer(
        devices=1,
        accelerator="gpu",
        max_epochs=max_epochs,
        precision="16-mixed",
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        logger=logger,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=10,
        # fast_dev_run=1,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
