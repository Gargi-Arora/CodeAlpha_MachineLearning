# Speech Emotion Recognition

Recognizes human emotions (happy, angry, sad, neutral, calm, fearful, disgust, surprised)
from speech audio using deep learning and audio signal processing.

## Pipeline
1. **Dataset**: [RAVDESS](https://zenodo.org/record/1188976) (download and unzip the
   `Audio_Speech_Actors_01-24.zip`, giving you `Actor_01/ ... Actor_24/` folders)
2. **Features**: MFCC, Chroma, Mel Spectrogram, Zero-Crossing Rate, Spectral Contrast
3. **Augmentation**: noise injection, pitch shift, time stretch (train split only)
4. **Models**: Random Forest (baseline) and a CNN + BiLSTM hybrid (main model)
5. **Demo**: Streamlit app for live predictions on uploaded audio

## Setup
```bash
pip install -r requirements.txt
```

## Train
```bash
python train.py /path/to/RAVDESS
```
This will:
- Build a manifest of all audio files and their emotion labels
- Extract features (with augmentation on the training split)
- Train and evaluate a Random Forest baseline and a CNN+BiLSTM model
- Save trained model, scaler, label encoder, confusion matrices, and
  training curves to `./artifacts/`

## Run the demo
```bash
streamlit run app.py
```
Upload a `.wav` file and get a predicted emotion with a confidence breakdown.

## Files
| File | Purpose |
|---|---|
| `feature_extraction.py` | Dataset parsing + audio feature extraction + augmentation |
| `train.py` | Model training and evaluation |
| `app.py` | Streamlit inference demo |
| `requirements.txt` | Python dependencies |

## Notes / things to mention in your report
- Train/val/test split is done **before** augmentation to avoid leakage
  (augmented copies of a sample never end up in val/test).
- Macro F1 is reported (not just accuracy) since it treats all emotion
  classes equally regardless of class size.
- Confusion matrices typically show the most confusion between
  `calm` vs `neutral` and `happy` vs `surprised` — worth discussing in
  your project report as a limitation/future work point.
