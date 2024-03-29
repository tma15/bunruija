# Bunruija
[![PyPI version](https://badge.fury.io/py/bunruija.svg)](https://badge.fury.io/py/bunruija)

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
  label_column: category
  text_column: title
  args:
    path: data/jsonl

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
  label_column: category
  text_column: title
  args:
    path: data/jsonl

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
  label_column: category
  text_column: title
  args:
    # Use local data in `data/jsonl`. In this path is assumed to contain data files such as train.jsonl, validation.jsonl and test.jsonl
    path: data/jsonl

    # If you want to use data on Hugging Face Hub, use the following args instead.
    # Data is from https://huggingface.co/datasets/shunk031/livedoor-news-corpus
    # path: shunk031/livedoor-news-corpus
    # random_state: 0
    # shuffle: true

```

data is loaded via [datasets.load_dataset](https://huggingface.co/docs/datasets/main/en/package_reference/loading_methods#datasets.load_dataset).
So, you can load local data as well as data on [Hugging Face Hub](https://huggingface.co/datasets).
When loading data, `args` are passed to `load_dataset`.

`label_column` and `text_column` are field names of label and text.

Format of `csv`:

```csv
category,sentence
sports,I like sports!
…
```

Format of `json`:

```json
[{"category", "sports", "text": "I like sports!"}]
```

Format of `jsonl`:

```json
{"category", "sports", "text": "I like suports!"}
```

### pipeline
You can set pipeline of your model in `pipeline` section.
It is a list of components that are used in your model.

For each component, `type` is a module path and `args` is arguments for the module.
For instance, when you set the first component as follows, [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) is instanciated with given arguments, and then applied to data at first in your model.

```yaml
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
```

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
