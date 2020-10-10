# Example usage of bunruija on Livedoor corpus

## Preparing data
You can download and decomporess dataset from the following:
```
wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
tar zxvf ldcc-20140209.tar.gz
```

After that, you can create csv files with the following:
```
python create_data /path/to/text
```

Then, `train.csv`, `dev.csv` and `test.csv` will be created at the current directory.
This script creates data for text classification where text is the title of a text and the category is the one of a text.

## Usage of bunruija
Bunruija can be used as follows:

```
bunruija-preprocess -y svm.yaml 
bunruija-train -y svm.yaml 
bunruija-evaluate -y svm.yaml 
```

As you see, all you have to do is to prepare datasets and a yaml file that contains all settings for text classification.
