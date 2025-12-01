"""
HW 3 CODE

Please write all your code for HW 3 in this file and include this file
in your submission.
"""
import csv
import sys
import time
from contextlib import contextmanager
from typing import Callable, Generator, Optional, Union

import nltk
from nltk.probability import FreqDist
from nltk.sentiment.util import split_train_test

# Type hints for emails and feature representations
Email = dict[str, Union[list[str], str]]
Features = dict[str, bool]

# This line of code is needed for reading large CSV files
csv.field_size_limit(sys.maxsize)

""" Code for measuring the time taken to run code """


def print_delay(*args, delay: float = .05, **kwargs):
    time.sleep(delay)
    print(*args, **kwargs)
    time.sleep(delay)


@contextmanager
def timer(message: str) -> Generator[Callable[[None], float], None, None]:
    """
    A timer that measures the time spent running code within a with-
    block.

    :param message: A message to be printed when the timer starts.
    :return: A generator that yields a function that returns the current
        time elapsed.
    """
    print_delay(message)

    # Start timer
    start_time = time.time()
    yield lambda: time.time() - start_time

    # Stop timer
    end_time = time.time()
    elapsed = end_time - start_time

    # Print time
    if elapsed >= 60:
        elapsed = f"{int(elapsed / 60)} minutes {elapsed % 60:.3f} seconds"
    else:
        elapsed = f"{elapsed:.3f} seconds"
    print_delay("Done. Time elapsed:", elapsed)


""" Code for Problem 1 """


def load_csv(filename: str) -> dict[str, list[str]]:
    """
    Problem 1b: Loads the data from a CSV file into Python. The data
    should be represented as a dict, where each column header (i.e.,
    the first row of each column) is a key, and the remaining rows are
    stored as a list.

    :param filename: The name of the csv file
    :return: A dict with the data in the csv file, in the following for-
        mat:
            {"col_header1": ["col1_row1", "col1_row2", ...],
             "col_header2": ["col2_row1", "col2_row2", ...],
             ...}
    """
    raise NotImplementedError("Problem 1b has not been completed yet!")


""" Code for Problem 2 """


def load_spam_data(filename: str) -> list[Email]:
    """
    Problem 2a: Loads the spam classification dataset from a .csv file.
    The type Email is defined at the top of this file, on line 18.

    :param filename: The name of the .csv file containing the dataset
    :return: The contents of the dataset, represented as a list of
        emails. Each email should be represented as a dict in the fol-
        lowing format:
            {"subject": ["The", "subject", "of", "the", "email"],
             "message": ["Hi", "Yulu", ",", "I", "need", "help", ...],
             "label": "ham"}
    """
    raise NotImplementedError("Problem 2a has not been completed yet!")


def get_vocab(emails: list[Email], part: str, n: Optional[int] = None) \
        -> set[str]:
    """
    Problem 2b: Extracts the top n most common token types in a list of
    emails.

    :param emails: A list of emails
    :param part: Which part of the emails to extract the top n most com-
        mon token types from ("subject" or "message")
    :param n: The number of token types to be extracted from each email.
        If n is none, then all token types should be extracted
    :return: The top n most common token types in either the subject or
        message of emails
    """
    raise NotImplementedError("Problem 2b has not been completed yet!")


def get_features(email: Email, subj_vocab: set[str], msg_vocab: set[str]) \
        -> Features:
    """
    Problem 2c: Converts an email into a dict of binary features. Each
    feature should be of the form "subject_contains(w)" where w is in
    subj_vocab, or "message_contains(w)" where w is in msg_vocab. The
    type Features is defined at the top of this file, on line 19.

    :param email: An email
    :param subj_vocab: The set of possible subject tokens
    :param msg_vocab: The set of possible message tokens
    :return: The email's features, represented in the following format:
        {"subject_contains(Re:)": True,
         "subject_contains(Fwd:)": False,
         "subject_contains(Hello)": True,
         ...,
         "message_contains(The)": True,
         "message_contains(of)": True,
         "message_contains(in)": False,
         ...}
    """
    raise NotImplementedError("Problem 2c has not been completed yet!")


class SpamClassifier:
    """
    A naive Bayes spam classifier that uses binary unigram features re-
    presenting subject tokens, message tokens, or both.
    """
    positive = "spam"

    def __init__(self, subj_vocab: set[str], msg_vocab: set[str]):
        """
        Constructs a SpamClassifier based on the set of possible unigram
        features for tokens in the subject and/or message of emails. To
        create a SpamClassifier that uses only subject tokens, simply
        set msg_vocab to be an empty set. To create a SpamClassifier
        that only uses message tokens, set subj_vocab to be an empty
        set.

        :param subj_vocab: The set of possible subject tokens
        :param msg_vocab: The set of possible message tokens
        """
        self.model = None
        if len(subj_vocab) + len(msg_vocab) == 0:
            raise ValueError("subj_vocab and msg_vocab can't both be empty!")
        self.subj_vocab = subj_vocab
        self.msg_vocab = msg_vocab

    """ Code for representing a SpamClassifier as a string """

    def name(self) -> str:
        if len(self.subj_vocab) == 0:
            return "Messages Only"
        elif len(self.msg_vocab) == 0:
            return "Subjects Only"
        else:
            return "Subjects + Messages"

    def __repr__(self) -> str:
        return f"SpamClassifier with {self.name().lower()}"

    def __str__(self) -> str:
        return self.name()

    """ Code for training and evaluating a SpamClassifier """

    def preprocess(self, emails: list[Email]) -> list[tuple[Features, str]]:
        """
        Problem 2d: Converts each email in a list of emails into a dict
            of features, paired with a classification label.

        :param emails: A dataset of emails
        :return: The dataset, in the following format:
            [({"subject_contains(Re)": True,
               "message_contains(the)": False,
               ...}, "spam"),
             ({"subject_contains(Re)": False,
               "message_contains(the)": True,
               ...}, "ham"),
             ...]
        """
        raise NotImplementedError("Problem 2d has not been completed yet!")

    def train(self, emails: list[Email]):
        """
        Trains the SpamClassifier on a list of emails.

        :param emails: The training dataset
        """
        dataset = self.preprocess(emails)
        self.model = nltk.NaiveBayesClassifier.train(dataset)

    def evaluate(self, emails: list[Email]) -> dict[str, float]:
        """
        Problem 2e: Calculates the accuracy, precision, recall, and F1
        score of this SpamClassifier on a list of emails. For precision,
        recall, and F1, the positive label is indicated by the class
        property SpamClassifier.positive.

        :param emails: A development or test set
        :return: The results of the evaluation, represented in the fol-
            lowing format:
                {"accuracy": .78,
                 "precision": .727,
                 "recall": .851,
                 "f1": .784}
        """
        if self.model is None:
            raise RuntimeError("This SpamClassifier has not been trained yet.")

        # Write your code here
        raise NotImplementedError("Problem 2e has not been completed yet!")


""" Script to train and test a SpamClassifier """

if __name__ == "__main__":
    # Load emails
    with timer("Loading data..."):
        all_emails = load_spam_data("data/enron_spam_data.csv")

        train, test = split_train_test(all_emails)
        train, dev = split_train_test(train)

    # Extract feature set
    with timer("Processing data..."):
        train_subj_vocab = get_vocab(train, "subject", n=500)
        train_msg_vocab = get_vocab(train, "message", n=2000)

    # Train some naïve Bayes classifiers
    models = [SpamClassifier(train_subj_vocab, train_msg_vocab),
              SpamClassifier(train_subj_vocab, set()),
              SpamClassifier(set(), train_msg_vocab)]

    for m in models:
        with timer(f"Training a {m.__repr__()}..."):
            m.train(train)

    # Evaluate models
    with timer("Evaluating models..."):
        # Evaluate on dev set
        dev_results = [m.evaluate(dev) for m in models]
        dev_f1 = [r["f1"] for r in dev_results]

        # Evaluate on test set
        best_model = models[dev_f1.index(max(dev_f1))]
        test_result = best_model.evaluate(test)

    # Report results
    print("\nMOST INFORMATIVE FEATURES")
    for m in models:
        print(f"\n{m}")
        m.model.show_most_informative_features(5)

    print("\nRESULTS")
    for m, r in zip(models, dev_results):
        print(f"{m} Dev: {r}")
    print(f"{best_model} Test: {test_result}")
