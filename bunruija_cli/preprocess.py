from bunruija import options
from bunruija.binarizer import Binarizer


def loader():
    import glob
    files = glob.glob(f'/Users/takuya/data/text/*/*.txt')
    categories = []
    titles = []
    for fname in files:
        elems = fname.split('/')
        category = elems[-2]
        with open(fname) as f:
            next(f)
            next(f)
            title = next(f).rstrip()
        categories.append(category)
        titles.append(title)
    print('#Files:', len(titles))
    return categories, titles


def train_dev_test_split(categories, titles):
    from sklearn.model_selection import train_test_split

    titles_train, titles_test, categories_train, categories_test = train_test_split(
        titles, categories, test_size=0.33, random_state=42)

    titles_train, titles_dev, categories_train, categories_dev = train_test_split(
        titles_train, categories_train, test_size=0.33, random_state=42)

    return (
        categories_train, titles_train,
        categories_dev, titles_dev,
        categories_test, titles_test
    )


def main():
    parser = options.get_default_preprocessing_parser()
    args = parser.parse_args()
    print(args)

    binarizer = Binarizer(args.yaml)
    binarizer.binarize(loader=loader, train_dev_test_split=train_dev_test_split)
