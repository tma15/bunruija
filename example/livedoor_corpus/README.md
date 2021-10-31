# Example usage of bunruija on Livedoor corpus

## Preparing data
You can download and decomporess dataset as following:
```
wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
tar zxvf ldcc-20140209.tar.gz  # text directory will be created
```

After that, you can create csv files:
```
python create_data.py /path/to/text
```

Then, `train.csv`, `dev.csv` and `test.csv` will be created in the current directory.
This script creates data for text classification where text is the title of a text and the category is the one of a text.

## Usage of bunruija
Bunruija can be used as follows:

```
bunruija-preprocess -y settings/svm.yaml 
bunruija-train -y settings/svm.yaml 
bunruija-evaluate -y settings/svm.yaml 
bunruija-predict -y settings/svm.yaml  # Input text and press Enter for obtaining a predicted label from a trained classifier.
```

As you see, all you have to do is to prepare datasets and a yaml file that contains all settings for text classification.


## Comparison of classification performance
Evaluation was conducted on Google Colaboratory (Colab).

|model|configuration file         |F1-score (%)|Average inference time on CPU (s)|Model size|pre-trained|
|-----|---------------------------|------------|---------------------------------|----------|-----------|
|BERT |`transformer.yaml`         | 87         |         0.022485                | 423M     | yes       |
|SVM  |`svm.yaml`                 | 82         |         0.002723                | 3.1M     | no        |
|QRNN |`qrnn.yaml`                | 80         |         0.016569                | 200M     | no        |
|PRADO|`prado.yaml`               | 79         |         0.023537                | 34M      | no        |
|LSTM |`lstm.yaml`                | 75         |         0.007223                | 587M     | no        |


Fine-tuning of classifiers other than SVM are conducted on a single GPU of Colab.
