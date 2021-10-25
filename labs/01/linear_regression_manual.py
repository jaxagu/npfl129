#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

# 0332e602-165f-11e8-9de3-00505601122b

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> float:
    # Load the Diabetes dataset
    dataset = sklearn.datasets.load_diabetes()

    # The input data are in `dataset.data`, targets are in `dataset.target`.

    # If you want to learn about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.
    print(dataset.DESCR)
    # TODO: Append a new feature to all input data, with value "1"
    diabetes_data = np.array(dataset.data)
    diabetes_data = np.hstack((np.ones((len(diabetes_data), 1), dtype=diabetes_data.dtype), diabetes_data))
    disease_progression = np.array(dataset.target)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(diabetes_data, disease_progression, test_size=args.test_size,
                                                                                random_state=args.seed)
    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).
    coeff = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train)),X_train.T),y_train)
    # TODO: Predict target values on the test set.
    prediction = np.dot(X_test,coeff)
    # TODO: Compute root mean square error on the test set predictions.
    rmse = np.sqrt(np.dot(np.subtract(prediction,y_test).T,np.subtract(prediction,y_test))/len(prediction))

    return rmse

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))
