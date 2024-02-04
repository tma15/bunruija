# Bunruija
Bunruija is a text classification toolkit.
Bunruija aims at enabling pre-processing, training and evaluation of text classification models with **minimum coding effort**.
Bunruija is mainly focusing on Japanese though it is also applicable to other languages.

See `example` for understanding how bunruija is easy to use.

## Features
- **Minimum requirements of coding**: bunruija enables users to train and evaluate their models through command lines. Because all experimental settings are stored in a yaml file, users do not have to write codes.
- **Easy to compare neural-based model with non-neural-based model**: because bunruija supports models based on scikit-learn and PyTorch in the same framework, users can easily compare classification accuracies and prediction times of neural- and non-neural-based models.
- **Easy to reproduce the training of a model**: because all hyperparameters of a model are stored in a yaml file, it is easy to reproduce the model.

## Install
```
pip install bunruija
```

## Example configs
Example of `sklearn.svm.SVC`

```yaml
data:
  train: train.csv
  dev: dev.csv
  test: test.csv

output_dir: models/svm-model

pipeline:
  - type: sklearn.feature_extraction.text.TfidfVectorizer
    args:
      tokenizer:
        type: bunruija.tokenizers.mecab_tokenizer.MeCabTokenizer
        args:
          lemmatize: true
          exclude_pos:
            - 助詞
            - 助動詞
      max_features: 10000
      min_df: 3
      ngram_range:
        - 1
        - 3
  - type: sklearn.svm.SVC
    args:
      verbose: false
      C: 10.
```

Example of BERT

```yaml
data:
  train: train.csv
  dev: dev.csv
  test: test.csv

output_dir: models/transformer-model

pipeline:
  - type: bunruija.feature_extraction.sequence.SequenceVectorizer
    args:
      tokenizer:
        type: transformers.AutoTokenizer
        args:
          pretrained_model_name_or_path: cl-tohoku/bert-base-japanese
  - type: bunruija.classifiers.transformer.TransformerClassifier
    args:
      device: cpu
      pretrained_model_name_or_path: cl-tohoku/bert-base-japanese
      optimizer:
        type: torch.optim.AdamW
        args:
          lr: 3e-5
          weight_decay: 0.01
          betas:
            - 0.9
            - 0.999
      max_epochs: 3
```

## CLI
```sh
# Training a classifier
bunruija-train -y config.yaml

# Evaluating the trained classifier
bunruija-evaluate -y config.yaml
```

## Config
### data
You can set data-related settings in `data`.

```yaml
data:
  train: train.csv  # training data
  dev: dev.csv # development data
  test: test.csv # test data
  label_column: label
  text_column: text
```

You can set local files in `train`, `dev`, and `test`.
Supported types are `csv`, `json` and `jsonl`.
`label_column` and `text_column` are field names of label and text.
When you set `label_column` to `label` and `text_column` to `text`, which are the default values, actual data must be as follows:

Format of `csv`:

```csv
label,text
label_name,sentence
…
```

Format of `json`:

```json
[{"label", "label_name", "text": "sentence"}]
```

Format of `jsonl`:

```json
{"label", "label_name", "text": "sentence"}
```

### pipeline
You can set pipeline of your model in `pipeline`


## Prediction using the trained classifier in Python code
After you trained a classification model, you can use that model for prediction as follows:
```python
from bunruija import Predictor

predictor = Predictor.from_pretrained("output_dir")
while True:
    text = input("Input:")
    label: list[str] = predictor([text], return_label_type="str")
    print(label[0])
```

`output_dir` is a directory that is specified in `output_dir` in config.
