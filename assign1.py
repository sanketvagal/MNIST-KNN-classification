from functools import cmp_to_key
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def knn_error_rates(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    k_min: int,
    k_max: int,
) -> List[float]:
    """Train the KNN model and calculate testing error rates for each K value

    Args:
        X_train (Union[pd.DataFrame, np.ndarray]): Training features data
        X_test (Union[pd.DataFrame, np.ndarray]): Testing features data
        y_train (Union[pd.Series, np.ndarray]): Training labels data
        y_test (Union[pd.Series, np.ndarray]): Testing labels data
        k_min (int): Minimum K value
        k_max (int): Maximum K value

    Returns:
        List[float]: List of testing error rates
    """
    # Array to store error rates for each K value between 1 and 20
    error_rates = []

    # Fitting the KNN model over each K value in range 1 and 20 inclusive
    for k in range(k_min, k_max + 1):
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        # Converting the error for each K value to percentages
        error = (1 - accuracy_score(y_test, y_pred)) * 100
        error_rates.append(error)

    return error_rates


def save(errors: List[float]) -> None:

    arr = np.array(errors)
    if len(arr.shape) > 2 or (len(arr.shape) == 2 and 1 not in arr.shape):
        raise ValueError(
            "Invalid output shape. Output should be an array " "that can be unambiguously raveled/squeezed."
        )
    if arr.dtype not in [np.float64, np.float32, np.float16]:
        raise ValueError("Your error rates must be stored as float values.")
    arr = arr.ravel()
    if len(arr) != 20 or (arr[0] >= arr[-1]):
        raise ValueError(
            "There should be 20 error values, with the first value " "corresponding to k=1, and the last to k=20."
        )
    if arr[-1] >= 2.0:
        raise ValueError(
            "Final array value too large. You have done something " "very wrong (probably relating to standardizing)."
        )
    if arr[-1] < 0.8:
        raise ValueError("You probably have not converted your error rates to percent values.")
    outfile = Path(__file__).resolve().parent / "errors.npy"
    np.save(outfile, arr, allow_pickle=False)


def plot_png(k_values: List[int], error_rates: List[float], img_name: str) -> None:
    """Plots the graph of K values versus the testing error rate.
        Save the graph as .png in the current working directory.

    Args:
        k_values (List[int]): List of K values to be used for X axis
        error_rates (List[float]): List of error rates in percentages to be used for Y axis
        img_name (str): Name of the image to be saved
    """
    # Plot the graph of k values vs the testing error rate and save the graph as a png using the provided values
    plt.plot(k_values, error_rates)
    plt.xlabel("K")
    plt.ylabel("Testing Error Rate in %")
    plt.savefig(img_name)


def process_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Perform preprocessing on the dataset and split it in features and labels

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and labels data
    """
    # Reading the dataset (XLS format)
    credit_default_dataset = pd.read_excel(Path(__file__).resolve().parent / "defaultofcreditcardclients.xls")

    # Replacing Header with Top Row
    credit_default_dataset.columns = credit_default_dataset.iloc[0]
    credit_default_dataset = credit_default_dataset[1:]

    # Dropping ID, as it is redundant, creating features data X
    # Dropping 'default payment next month' and making it labels y
    # Converting features and labels from object to int datatype, as sklearn cannot recognise object datatype

    X = credit_default_dataset.drop(["ID", "default payment next month"], axis=1).astype("int")
    y = (credit_default_dataset["default payment next month"]).astype("int")

    return X, y


def question1() -> None:
    """Plot the K vs testing error rate graph on the given data and save error rates."""
    # Loading the dataset
    numbers_dataset = loadmat(Path(__file__).resolve().parent / "NumberRecognitionAssignment1.mat")

    # Setting up Training and Testing Data, of 8s and 9s
    eights_train = numbers_dataset["imageArrayTraining8"]
    eights_test = numbers_dataset["imageArrayTesting8"]

    nines_train = numbers_dataset["imageArrayTraining9"]
    nines_test = numbers_dataset["imageArrayTesting9"]

    # Creating labels
    y_eights_train = np.zeros(eights_train.shape[-1])
    y_nines_train = np.ones(nines_train.shape[-1])

    y_eights_test = np.zeros(eights_test.shape[-1])
    y_nines_test = np.ones(nines_test.shape[-1])

    # Combining the 8s and 9s training data, along the count axis, i.e, 2
    X_train = np.concatenate((eights_train, nines_train), axis=2)
    X_test = np.concatenate((eights_test, nines_test), axis=2)

    # Combining the 8s and 9s test data
    y_train = np.concatenate((y_eights_train, y_nines_train), axis=0)
    y_test = np.concatenate((y_eights_test, y_nines_test), axis=0)

    # Swapping the number of samples feature to the 0th index
    X_train = X_train.transpose([2, 0, 1])
    X_test = X_test.transpose([2, 0, 1])

    # Reshaping the 3 dimensional array to 2 dimensional, required for KNeighborsClassifier
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

    k_min = 1
    k_max = 20
    k_values = list(range(k_min, k_max + 1))
    error_rates = knn_error_rates(X_train, X_test, y_train, y_test, k_min, k_max)
    plot_png(k_values, error_rates, img_name="knn_q1")

    save(error_rates)


def question2() -> None:
    """Calculate AUC values for the collected dataset, sorted according to separability."""

    def update_value(item: tuple) -> float:
        """Invert the values less than 0.5.

        Args:
            item (tuple): AUC value for the feature.
        Returns:
            float: Final AUC value for the feature.
        """
        if item[1] < 0.5:
            return float(1 - item[1])
        return float(item[1])

    def compare(item1: tuple, item2: tuple) -> int:
        """Compare two AUC values.

        Args:
            item1 (tuple): AUC value for first feature to be compared.
            item2 (tuple): AUC value for second feature to be compared.

        Returns:
            int: Key for comparing function
        """
        if update_value(item1) < update_value(item2):
            return -1
        elif update_value(item1) > update_value(item2):
            return 1
        else:
            return 0

    # Getting the features and target data for the dataset
    X, y = process_data()
    auc_values = {}

    # Calculating the ROC AUC score for each feature, rounding the score to 3 decimal places
    for column in X.columns:
        auc_values[column] = roc_auc_score(y, X[column]).round(3)

    # Sort the AUC values in terms of separability
    # Implemented a custom compare function:
    # https://stackoverflow.com/a/13239857
    auc_values_list = sorted(auc_values.items(), key=cmp_to_key(compare), reverse=True)

    # Print the top ten AUC values
    print(auc_values_list[:10])


def question3() -> None:
    """Plot the K vs testing error rate graph on the collected dataset."""
    # Getting the features and target data for the dataset
    X, y = process_data()

    # Splitting the data into training and testing sets
    # Turning off randomisation for reproducibility, enabling stratify to accomodate for imbalanced data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)

    k_min = 1
    k_max = 20
    k_values = list(range(k_min, k_max + 1))
    error_rates = knn_error_rates(X_train, X_test, y_train, y_test, k_min, k_max)
    plot_png(k_values, error_rates, img_name="knn_q3")


if __name__ == "__main__":
    question1()
    # question2()
    # question3()
