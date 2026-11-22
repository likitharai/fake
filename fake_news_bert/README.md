# Fake News Detection with BERT

A end-to-end NLP application that classifies news headlines and articles as **FAKE** or **REAL** using a BERT model (`bert-base-uncased`) fine-tuned on the WELFake dataset. It includes a training pipeline, evaluation, and a Streamlit dashboard for single-text detection, batch CSV scoring and model analytics.

## Project structure

```
fake_news_bert/
├── config.py               # all hyperparameters and paths
├── requirements.txt
├── README.md
├── .streamlit/config.toml  # light theme
├── src/
│   ├── preprocessing.py    # text cleaning, headline + body joining
│   ├── data.py             # loading, EDA, cleaning, dedupe, stratified split
│   ├── dataset.py          # PyTorch dataset + dynamic-padding collate
│   ├── model.py            # tokenizer/model loading, device selection (GPU/CPU)
│   ├── train.py            # fine-tuning, early stopping, test evaluation
│   ├── evaluate.py         # accuracy, precision, recall, F1, confusion matrix
│   └── predict.py          # FakeNewsPredictor inference API
├── app/
│   ├── streamlit_app.py    # entry point + sidebar navigation
│   ├── views.py            # Dashboard, Detection, Batch, Analytics, About
│   ├── services.py         # cached model and metric loaders
│   └── styles.py           # custom CSS and UI helpers
├── tests/test_pipeline.py  # unit tests (no model download required)
├── data/                   # place WELFake_Dataset.csv here
├── models/best_model/      # created by training
└── outputs/metrics/        # created by training (JSON artifacts for the app)
```

## 1. Installation

Python 3.9+ is required.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For GPU training, install the CUDA build of PyTorch from https://pytorch.org first. The code uses CUDA automatically when available and falls back to CPU.

## 2. Get the dataset

Download **WELFake_Dataset.csv** (search "WELFake" on Kaggle; it is described in Verma et al., *IEEE Transactions on Computational Social Systems*, 2021) and save it as:

```
data/WELFake_Dataset.csv
```

The file must contain `title`, `text` and `label` columns, where `label` is 0 = FAKE and 1 = REAL.

## 3. Train the model

```bash
python -m src.train
```

The pipeline:
1. Loads the data and runs EDA (missing values, duplicates, label balance, word counts), saved to `outputs/metrics/dataset_stats.json`.
2. Cleans text, drops empty rows and duplicates, and removes contradictory duplicates **before** splitting, so no article appears in more than one split.
3. Makes a stratified 70/15/15 train/validation/test split.
4. Fine-tunes BERT with AdamW, linear warm-up, gradient clipping and early stopping on validation loss.
5. Saves the best checkpoint and tokenizer to `models/best_model/`.
6. Evaluates that checkpoint once on the untouched test split and writes `outputs/metrics/test_metrics.json` and `training_history.json`.

**Quick CPU run:** full training on CPU is slow. In `config.py`, set `SAMPLE_SIZE = 20000` and `MAX_LENGTH = 128`. Metrics will be lower than the full run.

## 4. Run the tests

```bash
pytest -q
```

## 5. Launch the app

```bash
streamlit run app/streamlit_app.py
```

Open the URL shown in the terminal (default http://localhost:8501).

| Page | What it does |
|---|---|
| Dashboard | Project overview, model info, dataset statistics, metric cards |
| Fake News Detection | Single headline or article, verdict, confidence, probability chart, clear button |
| Batch Detection | Upload a CSV, pick text columns, get predictions, interactive table, CSV download |
| Model Analytics | Interactive confusion matrix, precision/recall/F1, loss and accuracy curves, classification report |
| About | Model description, workflow, tech stack, limitations |

The app displays only what the code produces: predictions come from the saved model, and metrics come from the JSON files written by training. If no model exists, the app says so instead of showing placeholder values.

## Design notes

- **No data leakage:** deduplication runs before splitting. The test set is used only once, after checkpoint selection on the validation set.
- **Positive class:** FAKE is the positive class for precision, recall and F1.
- **Efficient inference:** the model is loaded once and cached with `st.cache_resource`. Batches are tokenized with dynamic padding, and batch uploads are capped at `MAX_BATCH_ROWS`.
- **Device handling:** GPU is used when available, otherwise CPU.

## Limitations

- The model learns style and topical patterns from WELFake. It does not verify facts and may misclassify satire, opinion, breaking news or sources unlike the training data.
- Inputs are truncated to `MAX_LENGTH` tokens.
- Softmax confidence is not a calibrated probability of correctness.
- WELFake contains some label noise, which limits achievable accuracy.
