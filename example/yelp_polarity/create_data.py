import csv
import re

import datasets


def write(file_name, data):
    with open(file_name, "w") as f:
        writer = csv.writer(f)
        for sample in data:
            label = sample["label"]
            text = sample["text"]
            text = text.lower()
            text = re.sub("\.", " .", text)
            text = re.sub(",", " ,", text)
            text = re.sub("!", " !", text)
            text = re.sub("\?", " ?", text)
            writer.writerow([label, text])


def main():
    data = datasets.load_dataset("yelp_review_full")

    train_data = data["train"]
    write("train.csv", train_data)

    test_data = data["test"]
    write("test.csv", test_data)


if __name__ == "__main__":
    main()
