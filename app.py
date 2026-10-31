"""Streamlit demo of Chowk: classify a Kathmandu street vehicle into its PCU class, and replay the recorded sessions.

    streamlit run app.py

Needs the dataset, the detector and a champion classifier; if they are missing they are fetched
(checksummed) and trained once, like `make train` (about two minutes).
"""

import random

import pandas as pd
import streamlit as st
from PIL import Image

import brand
from chowk import config, data, features, model
from chowk.detector import Detector, zero_shot_class
from chowk.pipeline import replay_sessions
from chowk.store import LocalStore

brand.page("chowk")

CHAMPION = config.ARTIFACT_DIR / "models" / "champion"
NAMES = {
    "two_wheeler": "two-wheeler (motorbike, scooter)",
    "light": "light (car, taxi, jeep)",
    "medium": "medium (microbus, tempo, minibus)",
    "heavy": "heavy (bus, truck)",
}


@st.cache_resource(show_spinner="Fetching the dataset and detector and training the classifier (first visit only)...")
def load():
    data.download()
    if not (CHAMPION / "meta.json").exists():
        from argparse import Namespace

        from chowk.__main__ import cmd_train

        cmd_train(Namespace(bootstrap=True, version=None, no_augment=False))
    df, X = features.load()
    clf, meta = model.load(CHAMPION)
    return df, X, clf, meta, Detector(threads=2)


df, X, clf, meta, detector = load()


classify, replay = st.tabs(["Classify a vehicle", "Recorded sessions as 15-minute flows"])

with classify:
    st.write(
        "The camera box crops each vehicle the detector finds, and a small classifier on the detector's "
        "features puts it in a PCU class. An off-the-shelf detector's own guess (its COCO class) is shown "
        "for comparison."
    )
    c1, c2 = st.columns([1, 2])
    upload = c1.file_uploader("Upload a photo of one vehicle", type=["jpg", "jpeg", "png"])
    if c1.button("Or pick a random crop from the dataset") or "pick" not in st.session_state:
        st.session_state.pick = random.randrange(len(df))  # noqa: S311
    if upload:
        img, truth = Image.open(upload).convert("RGB"), None
    else:
        row = df.iloc[st.session_state.pick]
        with data.open_archive() as archive:
            img = data.load_image(archive, row["path"])
        truth = row["label"]
    c1.image(img, width=280)
    vec, coco = detector.crop_features(img)
    proba = clf.predict_proba(vec.reshape(1, -1))[0]
    best = clf.classes_[proba.argmax()]
    with c2:
        st.metric("Chowk's class", NAMES[best], help=f"{config.PCU[best]} PCU")
        st.write(
            f"**{config.PCU[best]} PCU** · confidence {proba.max():.0%}"
            + (
                " · low confidence, so the crop would go to the review queue"
                if proba.max() < config.MIN_CONFIDENCE
                else ""
            )
        )
        if truth:
            st.write(f"Hand label: **{NAMES[truth]}**" + (" ✅" if truth == best else " ❌"))
        zs = zero_shot_class(coco)
        st.write(f"Off-the-shelf detector alone: **{zs or 'nothing above threshold'}**")
        st.bar_chart(pd.Series(proba, index=[NAMES[c] for c in clf.classes_]), horizontal=True, x_label="probability")
    st.caption(
        f"Classifier {meta.version}. Note: dataset crops were used to train it, so these are in-sample; "
        "upload your own photo for a fair test."
    )

with replay:
    st.write("Every recorded session run through the counting code as 15-minute flows, with PCU per hour.")
    rows = replay_sessions(df, X, clf, meta.version, LocalStore(config.ARTIFACT_DIR / "app_flows.jsonl"))
    st.dataframe(rows, hide_index=True, width="stretch")
    for name in ("session_pcu.png", "confusion.png", "robustness.png"):
        p = config.ROOT / "docs" / "img" / name
        if p.exists():
            st.image(str(p))

brand.footer()
