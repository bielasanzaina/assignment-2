import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


def build_dataframe(folder):
    """
    Takes as input a directory containing presidential speeches and returns two
    DataFrames storing the text from those files, one for the training data
    and one for the test data (unlabeled).

    :param folder: a path to a directory containing presidential speeches
    :return: a tuple of pandas DataFrames
    """

    path = Path(folder)

    df_train = pd.DataFrame(columns=["author"])
    df_test = pd.DataFrame(columns=["author"])

    author_to_id_map = {
        "kennedy": 0,
        "johnson": 1
    }

    def make_df_from_dir(dir_name, df):
        """
        Takes as input directory to construct df from and returns updated df.

        :param dir_name: a Path to a directory
        :param df: an empty pandas DataFrame
        :return: updated pandas DataFrame
        """

        rows = []

        for f in path.glob(f"./{dir_name}/*.txt"):

            with open(f) as fp:
                text = fp.read()

                if dir_name in ("kennedy", "johnson"):

                    rows.append({
                        "author": dir_name,
                        "text": text
                    })

                else:

                    author = f.stem.split("_")[-1]

                    rows.append({
                        "author": author,
                        "text": text,
                        "filename": f.name
                    })

        if rows:
            df = pd.concat(
                [df, pd.DataFrame(rows)],
                ignore_index=True
            )

        return df

    for p in path.iterdir():

        if p.name in ("kennedy", "johnson"):
            df_train = make_df_from_dir(
                p.name,
                df_train
            )

        elif p.name == "unlabeled":
            df_test = make_df_from_dir(
                p.name,
                df_test
            )

    # Replace author names with numeric codes.
    df_train["author"] = df_train["author"].apply(
        lambda x: author_to_id_map.get(x)
    )

    df_test["author"] = df_test["author"].apply(
        lambda x: author_to_id_map.get(x)
    )

    return df_train, df_test


def train_nb(df, alpha=0.1):
    """
    Trains a Naive Bayes classifier by computing priors and likelihoods.

    :param df: a pandas DataFrame
    :param alpha: Lidstone smoothing parameter
    :return: vocabulary, priors, and likelihoods
    """

    vocabulary = {}

    for text in df["text"]:

        for token in text.split():

            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)

    n_docs = df.shape[0]
    n_classes = df["author"].nunique()

    # Compute prior probabilities.
    priors = np.zeros(shape=n_classes)

    for c in range(n_classes):

        priors[c] = (
            np.sum(df["author"] == c) / n_docs
        )

    # Create the bag-of-words training matrix.
    training_matrix = np.zeros(
        shape=(n_docs, len(vocabulary))
    )

    for doc_idx, text in enumerate(df["text"]):

        for token in text.split():

            training_matrix[
                doc_idx,
                vocabulary[token]
            ] += 1

    # Count words for each class.
    word_counts_per_class = np.zeros(
        shape=(n_classes, len(vocabulary))
    )

    for c in range(n_classes):

        class_docs = training_matrix[
            df["author"] == c
        ]

        word_counts_per_class[c] = np.sum(
            class_docs,
            axis=0
        )

    # Compute likelihoods using Lidstone smoothing.
    likelihoods = np.zeros(
        shape=(n_classes, len(vocabulary))
    )

    for c in range(n_classes):

        denominator = (
            np.sum(word_counts_per_class[c])
            + alpha * len(vocabulary)
        )

        for w in range(len(vocabulary)):

            likelihoods[c, w] = (
                word_counts_per_class[c, w] + alpha
            ) / denominator

    return vocabulary, priors, likelihoods


def test(df, vocabulary, priors, likelihoods):
    """
    Uses the custom Naive Bayes classifier to predict the author
    of each document.

    :param df: test DataFrame
    :param vocabulary: training vocabulary
    :param priors: class prior probabilities
    :param likelihoods: word likelihood matrix
    :return: list of predicted class labels
    """

    class_predictions = []

    for text in df["text"]:

        test_vector = np.zeros(
            shape=len(vocabulary)
        )

        for word in text.split():

            index = vocabulary.get(word)

            if index is not None:
                test_vector[index] += 1

        preds = np.log(priors) + np.sum(
            test_vector * np.log(likelihoods),
            axis=1
        )

        yhat = np.argmax(preds)

        class_predictions.append(yhat)

    return class_predictions


def sklearn_nb(training_df, test_df):
    """
    Performs Naive Bayes classification using scikit-learn.

    :param training_df: training data
    :param test_df: test data
    :return: predictions
    """

    vectorizer = CountVectorizer()

    # Learn vocabulary from the training data.
    training_data = vectorizer.fit_transform(
        training_df["text"]
    )

    # Transform test data using the same vocabulary.
    test_data = vectorizer.transform(
        test_df["text"]
    )

    nb_classifier = MultinomialNB()

    nb_classifier.fit(
        training_data,
        training_df["author"]
    )

    pred_nb = nb_classifier.predict(
        test_data
    )

    return pred_nb


def get_metrics(true, preds):
    """
    Computes accuracy, F1-score, and confusion matrix.

    :param true: DataFrame containing gold labels
    :param preds: predicted labels
    :return: accuracy, F1-score, confusion matrix
    """

    true_labels = true["author"]

    accuracy = metrics.accuracy_score(
        true_labels,
        preds
    )

    f1_score = metrics.f1_score(
        true_labels,
        preds
    )

    conf_matrix = metrics.confusion_matrix(
        true_labels,
        preds
    )

    return accuracy, f1_score, conf_matrix


def plot_confusion_matrix(custom_conf, sklearn_conf, labels):
    """
    Plots confusion matrices for the custom and
    scikit-learn classifiers.
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4)
    )

    sns.heatmap(
        custom_conf,
        annot=True,
        fmt="d",
        ax=axes[0]
    )

    axes[0].set_title(
        "Custom Naive Bayes"
    )
    axes[0].set_xlabel(
        "Predicted"
    )
    axes[0].set_ylabel(
        "True"
    )
    axes[0].set_xticklabels(
        labels
    )
    axes[0].set_yticklabels(
        labels
    )

    sns.heatmap(
        sklearn_conf,
        annot=True,
        fmt="d",
        ax=axes[1]
    )

    axes[1].set_title(
        "Scikit-learn Naive Bayes"
    )
    axes[1].set_xlabel(
        "Predicted"
    )
    axes[1].set_ylabel(
        "True"
    )
    axes[1].set_xticklabels(
        labels
    )
    axes[1].set_yticklabels(
        labels
    )

    plt.tight_layout()

    plt.savefig(
        "conf.jpg"
    )

    plt.show()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Naive Bayes Algorithm"
    )

    parser.add_argument(
        "-f",
        "--indir",
        required=True,
        help="Data directory"
    )

    args = parser.parse_args()

    # Build training and test DataFrames.
    training_df, test_df = build_dataframe(
        args.indir
    )

    # Train custom Naive Bayes classifier.
    vocabulary, priors, likelihoods = train_nb(
        training_df
    )

    # Predict test documents using custom classifier.
    class_predictions = test(
        test_df,
        vocabulary,
        priors,
        likelihoods
    )

    print("\nCustom Naive Bayes predictions:")

    for filename, pred in zip(
        test_df["filename"],
        class_predictions
    ):

        author = (
            "Kennedy"
            if pred == 0
            else "Johnson"
        )

        print(
            filename,
            "->",
            author
        )

    # Evaluate custom classifier.
    acc, f1, conf = get_metrics(
        test_df,
        class_predictions
    )

    print(
        "\nCustom accuracy:",
        acc
    )

    print(
        "Custom F1:",
        f1
    )

    print(
        "Custom confusion matrix:"
    )

    print(conf)

    # Run scikit-learn classifier.
    sklearn_preds = sklearn_nb(
        training_df,
        test_df
    )

    print(
        "\nScikit-learn predictions:"
    )

    for filename, pred in zip(
        test_df["filename"],
        sklearn_preds
    ):

        author = (
            "Kennedy"
            if pred == 0
            else "Johnson"
        )

        print(
            filename,
            "->",
            author
        )

    # Evaluate scikit-learn classifier.
    sklearn_acc, sklearn_f1, sklearn_conf = get_metrics(
        test_df,
        sklearn_preds
    )

    print(
        "\nScikit-learn accuracy:",
        sklearn_acc
    )

    print(
        "Scikit-learn F1:",
        sklearn_f1
    )

    print(
        "Scikit-learn confusion matrix:"
    )

    print(sklearn_conf)

    # Plot both confusion matrices.
    plot_confusion_matrix(
        conf,
        sklearn_conf,
        [0, 1]
    )