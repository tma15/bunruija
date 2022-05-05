import csv
import sys
import glob

from sklearn.model_selection import train_test_split


def loader(data_dir):
    files = glob.glob(f"{data_dir}/*/*.txt")
    categories = []
    titles = []
    for fname in files:
        elems = fname.split("/")
        category = elems[-2]
        with open(fname) as f:
            next(f)
            next(f)
            title = next(f).rstrip()
        categories.append(category)
        titles.append(title)
    print("#Files:", len(titles))
    return categories, titles


def train_dev_test_split(categories, titles):
    titles_train, titles_test, categories_train, categories_test = train_test_split(
        titles, categories, test_size=0.33, random_state=42
    )

    titles_train, titles_dev, categories_train, categories_dev = train_test_split(
        titles_train, categories_train, test_size=0.33, random_state=42
    )

    return (
        categories_train,
        titles_train,
        categories_dev,
        titles_dev,
        categories_test,
        titles_test,
    )


def write_csv(categories, titles, name):
    with open(name, "w") as f:
        writer = csv.writer(f)
        for category, title in zip(categories, titles):
            writer.writerow([category, title])


def main():
    categories, titles = loader(sys.argv[1])

    (
        categories_train,
        titles_train,
        categories_dev,
        titles_dev,
        categories_test,
        titles_test,
    ) = train_dev_test_split(categories, titles)

    write_csv(categories_train, titles_train, "train.csv")
    write_csv(categories_dev, titles_dev, "dev.csv")
    write_csv(categories_test, titles_test, "test.csv")


if __name__ == "__main__":
    main()
