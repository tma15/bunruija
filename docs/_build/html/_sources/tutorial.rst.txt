Tutorial
====================================

================
Data file format
================

.. code-block:: sh

   label1,text
   label2,text
   label2,text
   label3,text
   label4,text

=================
Model file format
=================


.. code-block:: yaml

  preprocess:
    data:
      train: train.csv
      dev: dev.csv
      test: test.csv
  
  tokenizer:
    type: mecab
    args:
      lemmatize: true
      exclude_pos:
        - 助詞
        - 助動詞
  
  bin_dir: models/svm-model
  
  classifier:
    - type: tfidf
      args:
        max_features: 10000
        min_df: 3
        ngram_range:
          - 1
          - 3
    - type: svm
      args:
        verbose: false
        C: 10.


.. code-block:: sh

   bunruija-preprocess -y settings/model.yaml
   bunruija-train -y settings/model.yaml
   bunruija-evaluate -y settings/model.yaml
