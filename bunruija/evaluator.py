import csv
from pathlib import Path
import pickle
import time

import sklearn
import yaml

import bunruija
from bunruija.predictor import Predictor


class Evaluator:
    """Evaluates a trained model
    """
    def __init__(self, args):
        with open(args.yaml) as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)

        self.evaluate_time = not args.no_evaluate_time
        self.verbose = args.verbose

        self.predictor = Predictor(args.yaml)

        with open(Path(self.config.get('bin_dir', '.')) / 'data.bunruija', 'rb') as f:
            self.data = pickle.load(f)

    def evaluate(self):
        if self.evaluate_time:
            with open(self.config.get('preprocess', {}).get('data', {}).get('test', '')) as f:
                reader = csv.reader(f)
                y_pred = []
                y_test = []
                start_at = time.perf_counter()
                n = 0
                for row in reader:
                    y_pred_i = self.predictor(row[1])
                    y_pred.append(y_pred_i)
                    y_test.append(row[0])
                    n += 1
                duration = (time.perf_counter() - start_at) / n
        else:
            X_test = self.data['data_test']
            y_test = self.data['label_test']
            y_pred = self.predictor(X_test)

        labels = list(self.predictor.label_encoder.classes_)
        if self.verbose:
            conf_mat = sklearn.metrics.confusion_matrix(y_test, y_pred)
            for i in range(len(labels)):
                print('True', 'Pred', 'Num samples', sep='\t')
                for j in range(len(labels)):
                    print(labels[i], labels[j], conf_mat[i, j], sep='\t')
                print()

        report = sklearn.metrics.classification_report(y_test, y_pred, target_names=labels)
        print(report)
        if self.evaluate_time:
            print(f'Average prediction time: {duration} sec.')
