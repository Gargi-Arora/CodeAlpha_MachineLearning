"""
feature_extraction.py
----------------------
Utilities to:
  1. Walk the RAVDESS dataset folder and build a (filepath, emotion) dataframe
  2. Extract audio features (MFCC, Chroma, Mel Spectrogram, ZCR, Spectral Contrast)
  3. Apply simple data augmentation (noise, pitch shift, time stretch)

RAVDESS filename format example: 03-01-06-01-02-01-12.wav
Position 3 (index 2) = emotion code:
  01 = neutral, 02 = calm, 03 = happy, 04 = sad,
  05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
"""

import os
import glob
import numpy as np
import pandas as pd
import librosa

RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

SAMPLE_RATE = 22050
DURATION = 3.0          # seconds - clips are padded/truncated to this length
N_MFCC = 40


def build_dataframe(ravdess_root: str) -> pd.DataFrame:
    """
    Scan the RAVDESS root directory (containing Actor_01, Actor_02, ... folders)
    and return a dataframe with columns: filepath, emotion
    """
    rows = []
    wav_files = glob.glob(os.path.join(ravdess_root, "Actor_*", "*.wav"))
    if not wav_files:
        raise FileNotFoundError(
            f"No .wav files found under {ravdess_root}. "
            "Check that you passed the correct RAVDESS root folder."
        )
    for path in wav_files:
        filename = os.path.basename(path)
        parts = filename.split("-")
        emotion_code = parts[2]
        emotion = RAVDESS_EMOTION_MAP.get(emotion_code)
        if emotion is None:
            continue
        rows.append({"filepath": path, "emotion": emotion})
    df = pd.DataFrame(rows)
    print(f"Found {len(df)} audio files across {df['emotion'].nunique()} emotions.")
    print(df["emotion"].value_counts())
    return df


def _pad_or_truncate(y: np.ndarray, sr: int, duration: float = DURATION) -> np.ndarray:
    target_len = int(sr * duration)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, max(0, target_len - len(y))), mode="constant")
    return y


def add_noise(y: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


def pitch_shift(y: np.ndarray, sr: int, n_steps: float = 2.0) -> np.ndarray:
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def time_stretch(y: np.ndarray, rate: float = 1.1) -> np.ndarray:
    return librosa.effects.time_stretch(y, rate=rate)


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extract and concatenate: MFCC (mean over time), Chroma, Mel Spectrogram,
    Zero-Crossing Rate, and Spectral Contrast. Returns a 1D feature vector.
    """
    y = _pad_or_truncate(y, sr)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    stft = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel.T, axis=0)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr.T, axis=0)

    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)

    return np.concatenate(
        [mfcc_mean, chroma_mean, mel_mean, zcr_mean, contrast_mean]
    )


def extract_features_from_file(filepath: str, augment: bool = False) -> list:
    """
    Load a file and return a list of feature vectors.
    If augment=True, also returns features for noise/pitch/stretch versions
    (useful only for the TRAIN split - never augment val/test data).
    """
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE)
    feature_list = [extract_features(y, sr)]

    if augment:
        feature_list.append(extract_features(add_noise(y), sr))
        try:
            feature_list.append(extract_features(pitch_shift(y, sr), sr))
        except Exception:
            pass
        try:
            feature_list.append(extract_features(time_stretch(y), sr))
        except Exception:
            pass

    return feature_list


def build_feature_dataset(df: pd.DataFrame, augment: bool = False):
    """
    Given a dataframe with 'filepath' and 'emotion' columns, extract features
    for every file (optionally augmented) and return X (features) and y (labels).
    """
    X, y_labels = [], []
    for i, row in df.iterrows():
        feats = extract_features_from_file(row["filepath"], augment=augment)
        for f in feats:
            X.append(f)
            y_labels.append(row["emotion"])
        if i % 50 == 0:
            print(f"Processed {i}/{len(df)} files...")
    return np.array(X), np.array(y_labels)


if __name__ == "__main__":
    # Example usage:
    #   python feature_extraction.py /path/to/RAVDESS
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "./RAVDESS"
    df = build_dataframe(root)
    df.to_csv("ravdess_manifest.csv", index=False)
    print("Saved manifest to ravdess_manifest.csv")
