import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
from glob import glob
import cv2
import pandas as pd
import seaborn as sns

if st.checkbox("Debug datasets_pkl"):
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

st.write(sorted(images_train)[:samples_to_show])

df = pd.DataFrame({"fname": images_train})
df["id"] = df.fname.apply(lambda x: x.split("/")[-1].split("_")[0])
df["camera"] = df.fname.apply(lambda x: x.split("/")[-1].split("_")[1])
df["sample"] = df.fname.apply(lambda x: x.split("/")[-1].split("_")[1])


st.write(df)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))
sns.histplot(df.id.value_counts(), ax=ax0)
ax0.set_xlabel("Samples count per id")
ax0.set_ylabel("Popularity\ncount of ids with the same sample count")

sns.histplot(df.id.value_counts(), ax=ax1)
ax1.set_yscale("log")
ax1.set_xlabel("Samples count per id")
ax1.set_ylabel("Popularity\ncount of ids with the same sample count")
st.pyplot(fig)


if st.checkbox("Show some samples"):

    for fname in sorted(images_train)[:samples_to_show]:
        st.text(fname)
        img = cv2.imread(fname)[:, :, ::-1]

        st.image(img)


vc = df.id.value_counts()
id_with_max_samples = list(dict(vc[np.argmax(vc) : np.argmax(vc) + 1]).keys())[0]


if st.checkbox("Show samples with maximum count"):
    max_samples_df = df[df.id == id_with_max_samples]

    fig = plt.figure()
    max_samples_df.camera.value_counts().plot.barh()
    st.pyplot(fig)

    for fname in sorted(max_samples_df.fname):
        st.text(fname)
        img = cv2.imread(fname)[:, :, ::-1]

        st.image(img)
