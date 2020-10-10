from pathlib import Path
import pickle
import yaml

import sklearn

import bunruija


class Evaluator:
    def __init__(self, config_file):
        with open(config_file) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        with open(Path(config.get('bin_dir', '.')) / 'model.bunruija', 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['classifier']
            self.label_encoder = model_data['label_encoder']
            self.vectorizer = model_data['vectorizer']

            tokenizer_name = model_data['tokenizer']
            tokenizer = getattr(bunruija.tokenizers, tokenizer_name)()
            self.vectorizer.tokenizer = tokenizer

        with open(Path(config.get('bin_dir', '.')) / 'data.bunruija', 'rb') as f:
            self.data = pickle.load(f)

    def evaluate(self):
        y_test = self.data['label_test']
        X_test = self.data['data_test']
        y_pred = self.model.predict(X_test)

        conf_mat = sklearn.metrics.confusion_matrix(y_test, y_pred)
        labels = list(self.label_encoder.classes_)
        for i in range(len(labels)):
            print('True', 'Pred', 'Num samples', sep='\t')
            for j in range(len(labels)):
                print(labels[i], labels[j], conf_mat[i, j], sep='\t')
            print()

        fscore = sklearn.metrics.f1_score(y_test, y_pred, average='micro')
        print(f'F-score: {fscore}')
