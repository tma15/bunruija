# Evaluation Results

## Linear SVM
### Config
```json
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
F-score on dev: 0.7514450867052023
              precision    recall  f1-score   support

  acceptable       0.86      0.85      0.85       733
unacceptable       0.20      0.21      0.21       132

   accuracy                            0.75       865
   macro avg       0.53      0.53      0.53       865
weighted avg       0.76      0.75      0.75       865
```
