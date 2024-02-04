# Evaluation Results

## Linear SVM
### Config
```yaml
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
  - type: sklearn.svm.LinearSVC
    args:
      verbose: 10
      C: 10.
```

### Results
```
F-score on dev: 0.9225327201980899
              precision    recall  f1-score   support

    negative       0.56      0.85      0.68       542
    positive       0.98      0.93      0.96      5112

    accuracy                           0.92      5654
   macro avg       0.77      0.89      0.82      5654
weighted avg       0.94      0.92      0.93      5654
```
