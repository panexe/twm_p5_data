# Just a dump of all utility classes and funitions to make the notebook more lightweight
from typing import Any, Union, Dict, List, Tuple
import os
import urllib
import subprocess
from io import StringIO
from zipfile import ZipFile
import pydotplus
from tqdm import tqdm
import ipywidgets as widgets
from IPython.display import (
    Markdown,
    HTML,
    display_markdown,
    display_html,
    Image,
    display,
    Javascript,
)
from sklearn.tree import export_graphviz
import seaborn as sns

import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import pandas as pd


# function for widget interaction
def f(epochs):
    return epochs


def scale_min_max(array):
    return (array - min(array)) / (max(array) - min(array))


# -----------------------------------------------#
#             GloVe Embeddings
# -----------------------------------------------#


# https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
# https://github.com/tqdm/tqdm#hooks-and-callbacks
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_glove_embeddings() -> Any:
    """Downloads GloVe embeddings from a web link and saves it to the local file system.

    Returns:
        None
    """
    glove6B = "glove.6B.50d.txt"

    if os.path.isfile(glove6B):
        print("Glove tokens file already exists. No need to download it.")
    else:
        print("Glove tokens file does not exists. Download can take some time.")
        download_url("http://nlp.stanford.edu/data/glove.6B.zip", "glove.6B.zip")
        ZipFile("glove.6B.zip").extract(glove6B)
        print("Done!")


def load_glove_embeddings(
    filepath: str = "glove.6B.50d.txt", verbose: Union[int, bool] = 1
) -> Dict[str, np.ndarray]:
    """Loads GloVe embeddings from a text file and returns a dictionary mapping words to their embeddings.

    Arguments:
    - filepath: path to glove embeddings.
    - verbose: level of verbosity. Default is 1.

    Returns:
        A dictionary mapping words to their embeddings as numpy arrays.
    """
    with open(filepath, "r") as file:
        embeddings = file.readlines()

    # Create a dictionary to map words to their embeddings
    word_embeddings = {}
    for line in embeddings:
        values = line.split()
        word = values[0]
        embedding = np.asarray(values[1:], dtype="float32")
        word_embeddings[word] = embedding

    if verbose:
        print(f"Loaded {len(word_embeddings)} embeddings.")

    return word_embeddings


def create_embeddings_matrix(
    vocab_size: int,
    word_index: Dict[str, int],
    word_embeddings: Dict[str, np.ndarray],
    verbose: Union[int, bool] = 1,
) -> np.ndarray:
    """Creates an embeddings matrix to use in a keras embeddings layer.

    Arguments:
    - vocab_size: amount of words embedded
    - word_index: dict mapping containing all words to be emedded as items
    - word_embeddings: dict mapping words to vectors
    - verbose: level of verbosity. Default is 1.

    Returns:
        numpy.ndarray containing the embedding vectors.
    """

    # Create a matrix of zeros with the correct shape
    glove_weights = np.zeros((vocab_size, 50))

    hits = 0
    # Map the words in the vocabulary to their corresponding embeddings
    for word, index in word_index.items():
        embedding_vector = word_embeddings.get(word)
        if index >= vocab_size:
            continue
        if embedding_vector is not None:
            glove_weights[index] = embedding_vector
            hits += 1
    if verbose:
        print(f"Matched {hits} out of {vocab_size} words.")

    return glove_weights


# -----------------------------------------------#
#             VISUALIZATIONS
# -----------------------------------------------#


def check_graphviz_installation(verbose=1):
    "Checks if graphviz is installed and prints the result to the screen. Also returns true or false."

    child = subprocess.Popen(
        ["dot", "-v"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    graphviz_test_output = child.stdout.read().decode().split("\n")

    graphviz_is_installed = False
    for line in graphviz_test_output:
        if "dot - graphviz version" in line:
            graphviz_is_installed = True
            break
    if not verbose:
        return graphviz_is_installed
    if graphviz_is_installed:
        print("Graphviz ist auf dem System installiert.")
        return True
    print(
        "Graphviz scheint nicht installiert zu sein. Es kann mit: `(sudo) apt install graphviz -y` installiert werden."
    )
    return False


def create_set_value_ui(default_vocab_size, default_seq_len, default_cached, callback_fn):
    """Creates a jupyter widgets ui"""

    auto_layout = widgets.Layout(display="flex", width="auto")

    box_layout = widgets.Layout(
        display="flex",
        flex_flow="column",
        justify_content="space-between",
        align_items="stretch",
        border="none",
        width="50%",
    )

    vocab_dd = widgets.Dropdown(
        options=["1000", "5000", "10000", "15000", "20000"],
        value="10000",
        disabled=False,
    )

    seq_dd = widgets.Dropdown(
        options=["100", "250", "500"],
        value="250",
        disabled=False,
    )

    cached_cb = widgets.Checkbox(False)

    vocab_box = widgets.HBox(
        [widgets.Label("Wörteranzahl:", layout=widgets.Layout(width="50%")), vocab_dd],
        layout=auto_layout,
    )
    seq_box = widgets.HBox(
        [widgets.Label("Sequenzlänge:", layout=widgets.Layout(width="50%")), seq_dd],
        layout=auto_layout,
    )
    cached_box = widgets.HBox(
        [widgets.Label("Cached:", layout=widgets.Layout(width="auto")), cached_cb],
        layout=auto_layout,
    )
    submit_button = widgets.Button(description="Set Values", layout=auto_layout)

    
    submit_button.on_click(callback_fn(seq_dd, vocab_dd, cached_cb))

    items = [vocab_box, seq_box, cached_box, submit_button]
    box = widgets.Box(children=items, layout=box_layout)

    display(box)


def print_md(md_string: str):
    """Displays text in a Jupyter Notebook.

    Args:
    - md_string: A string that may contain markdown format.

    Returns:
    - None
    """
    display_markdown(Markdown(md_string))


def display_table(table_data, use_header: bool = True):
    """Displays a table in a Jupyter Notebook.

    Args:
    - table_data: A list of rows, where each row is a list of values.
    - use_header: A boolean indicating whether the first row should be used as the table header.

    Returns:
    - None
    """
    num_columns = len(table_data[0])
    for row in table_data:
        assert len(row) == num_columns, "All rows must have the same number of columns"

    if use_header:
        table_header = "<tr><th>" + "</th><th>".join(table_data[0]) + "</th></tr>"
        table_rows = "</tr><tr>".join(
            "<td>" + "</td><td>".join(str(_) for _ in row) + "</td>"
            for row in table_data[1:]
        )
    else:
        table_header = ""
        table_rows = "</tr><tr>".join(
            "<td>" + "</td><td>".join(str(_) for _ in row) + "</td>"
            for row in table_data
        )

    display_html(
        HTML(
            '<table style="text-align: left"><tr>{}{}</tr></table>'.format(
                table_header, table_rows
            )
        )
    )


def visualize_classification_report(classification_report_data, class_names):
    """Displays a table in a Jupyter Notebook.

    Args:
    - classification_report_data: Data returned by sklearn classification_report.
    - class_names: Names of the predicted classes.

    Returns:
    - None
    """
    precision = [
        classification_report_data[cat]["precision"]
        for cat in classification_report_data
        if cat != "accuracy" and cat != "macro avg" and cat != "weighted avg"
    ]
    recall = [
        classification_report_data[cat]["recall"]
        for cat in classification_report_data
        if cat != "accuracy" and cat != "macro avg" and cat != "weighted avg"
    ]
    f1_score = [
        classification_report_data[cat]["f1-score"]
        for cat in classification_report_data
        if cat != "accuracy" and cat != "macro avg" and cat != "weighted avg"
    ]


    # Set the bar width
    bar_width = 0.25

    # Set the bar positions
    r1 = np.arange(len(precision))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Make the plot
    plt.bar(r1, precision, width=bar_width, label="Precision")
    plt.bar(r2, recall, width=bar_width, label="Recall")
    plt.bar(r3, f1_score, width=bar_width, label="F1 Score")

    # Add labels and tick marks
    plt.xticks([r + bar_width for r in range(len(precision))], class_names)
    plt.ylabel("Metric Value")
    plt.legend()

    # Show the plot
    plt.show()


def list_decision_tree_split_arguments(model):
    """Lists the split arguments of a sklearn decision tree model.

    Arguments:
    - model: A sklearn decision treee model

    Returns:
    - None
    """
    splitArgs = model.named_steps["tree"].tree_.feature.tolist()

    elements = ["\n**Das sind die Splitargumente des DecisionTrees:**\n"]

    for i, x in tqdm(enumerate([arg for arg in splitArgs if arg != -2])):
        rep = model.named_steps["tfidf"].get_feature_names_out()[x]
        elements.append(f"{i+1}. {rep}\n")
    print_md("".join(elements))


def display_decision_tree(model):
    """Display a sklearn decision tree model.

    **Requires graphviz to be installed!**

    Returns:
    - None
    """
    dot_data = StringIO()

    export_graphviz(
        model.named_steps["tree"],
        out_file=dot_data,
        filled=True,
        rounded=True,
        special_characters=True,
    )

    splitArgs = model.named_steps["tree"].tree_.feature.tolist()

    dot_data = dot_data.getvalue()

    for x in [arg for arg in splitArgs if arg != -2]:
        rep = model.named_steps["tfidf"].get_feature_names_out()[x]
        dot_data = dot_data.replace("x<SUB>" + str(x) + "</SUB>", str(rep))

    graph = pydotplus.graph_from_dot_data(dot_data)
    display(Image(graph.create_png()))


def plot_training_history(history):
    """Plots the loss and binary accuracy of a training run.

    Arguments:
    - history: history returned by calling .fit() on a keras model.

    Returns:
    - None
    """

    # Extract the training and validation data from the history object
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    train_acc = history.history["binary_accuracy"]
    val_acc = history.history["val_binary_accuracy"]

    # Create a dataframe from the training and validation data
    df_loss = pd.DataFrame({"train_loss": train_loss, "val_loss": val_loss})
    df_acc = pd.DataFrame({"train_acc": train_acc, "val_acc": val_acc})

    # Set up the plot figure and axes
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # Use Seaborn to create line plots of the training and validation loss and accuracy
    sns.lineplot(data=df_loss, palette="tab10", linewidth=2.5, ax=ax1)
    sns.lineplot(data=df_acc, palette="tab10", linewidth=2.5, ax=ax2)

    # Set the plot titles
    ax1.set_title("Training and Validation Loss")
    ax2.set_title("Training and Validation Accuracy")


# https://developers.google.com/machine-learning/guides/text-classification/step-2
# "sepcnn"


def plot_sample_length_distribution(sample_texts):
    """Plots the sample length distribution.

    # Arguments
        samples_texts: list, sample texts.
    """
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel("Length of a sample")
    plt.ylabel("Number of samples")
    plt.title("Sample length distribution")
    plt.show()


# -----------------------------------------------#
#             dataset.py
# -----------------------------------------------#


class IMDbDataset:
    CLASS_NAMES = ["Negativ", "Positiv"]

    def __init__(
        self, vocabulary_size: int, verbose: int = 1, preprocess_data: bool = False
    ):
        """
        A wrapper around the tensorflow imdb dataset wrapper.

        Usage example:
        ```
        dataset = IMDbDataset(10_000)
        (X_train, y_train), (X_test, y_test) = dataset.get_data()
        (X_train_decoded, _), (X_test_decoded, _) = dataset.get_data(True)
        ```

        Parameters:
        - vocabulary_size: The number of most common words to consider in the dataset.
        - verbose: An integer indicating the verbosity level. 0 means no output, 1 means some output, and 2 means more output.
        - preprocess_data: A boolean indicating whether to preprocess the data before using it.
        """
        # Config
        self._verbose = verbose
        self._vocabulary_size = vocabulary_size
        self._preprocess_data = preprocess_data

        # Load data
        if self._verbose > 0:
            print("Loading data...")
        (self.X_train, self.y_train), (
            self.X_test,
            self.y_test,
        ) = tf.keras.datasets.imdb.load_data(num_words=self._vocabulary_size)

        # Load word index and reverse word index
        if self._verbose > 0:
            print("Loading word index and reverse word index...")
        self.word_index = tf.keras.datasets.imdb.get_word_index()
        self._reverse_word_index = dict(
            (i, word) for (word, i) in self.word_index.items()
        )

        # Decode sequences in parallel
        if self._verbose > 0:
            print("Decoding sequences...")
        self.X_train_decoded = list(
            tqdm(map(self.decode_sequence, self.X_train), total=len(self.X_train))
        )
        self.X_test_decoded = list(
            tqdm(map(self.decode_sequence, self.X_test), total=len(self.X_test))
        )

        # Convert lists to arrays
        if self._verbose > 0:
            print("Converting lists to arrays...")
        self.X_train_decoded = np.array(self.X_train_decoded, dtype=object)
        self.X_test_decoded = np.array(self.X_test_decoded, dtype=object)

        # Preprocess data if requested
        if self._preprocess_data:
            if self._verbose > 0:
                print("Preprocessing data...")
            self.X_train_decoded = self._preprocess(self.X_train_decoded)
            self.X_test_decoded = self._preprocess(self.X_test_decoded)

        if self._verbose > 0:
            print("Done.")

    def decode_sequence(self, sequence: List[int]) -> List[str]:
        """
        Decode a sequence of integers into a list of words.

        Parameters:
        - sequence: A list of integers representing a sequence of words.

        Returns:
        - A list of strings representing the decoded sequence of words
        """
        return [self._reverse_word_index.get(i - 3, "?") for i in sequence]

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess the data.

        Parameters:
        - data: A numpy array containing the data to preprocess.

        Returns:
        - A numpy array containing the preprocessed data.
        """
        # TODO: Add preprocessing, removing br tokens
        return data

    def get_data(
        self, string_representation: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the training and test data.

        Returns:
        - A tuple containing the training data (X_train, y_train) and the test data (X_test, y_test).
        """
        if string_representation:
            return (self.X_train_decoded, self.y_train), (
                self.X_test_decoded,
                self.y_test,
            )

        return (self.X_train, self.y_train), (self.X_test, self.y_test)

    def decode_sequence(self, sequence: List[int]) -> str:
        """
        Decode a sequence of integers into a string of words.

        Parameters:
        - sequence: A list of integers representing a sequence of words.

        Returns:
        - A string representing the decoded sequence of words.
        """
        return " ".join(self._reverse_word_index.get(i - 3, "?") for i in sequence)

    def print_metadata(self):
        """
        Calculate and print some basic metadata about the dataset.

        Returns:
        - None
        """
        output = [["Trainingsdaten Metrik", "Wert"]]
        output.append(["Anzahl an Samples", f"{self.y_train.shape[0]}"])

        classes = np.unique(self.y_train)
        output.append(["Samples pro Klasse:", ""])
        for c in classes:
            output.append(
                [f"{IMDbDataset.CLASS_NAMES[c]}", f"{np.sum(self.y_train == c)}"]
            )
        n_words = [len(x) for x in self.X_train]
        output.append(
            ["Durschnitt: Wörter pro Sample", f'{"%.3f" % (np.mean(n_words))}']
        )
        output.append(["Median: Wörter pro Sample", f"{np.median(n_words)}"])
        display_table(output)

        output = [["Testdaten Metrik", "Wert"]]
        output.append(["Anzahl an Samples", f"{self.y_test.shape[0]}"])
        output.append(["Samples pro Klasse:", ""])
        for c in classes:
            output.append(
                [f"{IMDbDataset.CLASS_NAMES[c]}", f"{np.sum(self.y_test == c)}"]
            )
        n_words = [len(x) for x in self.X_test]
        output.append(
            ["Durschnitt: Wörter pro Sample", f'{"%.3f" % (np.mean(n_words))}']
        )
        output.append(["Median: Wörter pro Sample", f"{np.median(n_words)}"])
        display_table(output)
