#!/usr/bin/env python3
import argparse
from math import e

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

# 0332e602-165f-11e8-9de3-00505601122b

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="diabetes", type=str, help="Standard sklearn dataset to load")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(dataset.data, dataset.target,
                                                                                    test_size=args.test_size,
                                                                                    random_state=args.seed)

    # TODO: Process the input columns in the following way:
    #
    # - if a column has only integer values, consider it a categorical column
    #   (days in a week, dog breed, ...; in general, integer values can also
    #   represent numerical non-categorical values, but we use this assumption
    #   for the sake of an exercise). Encode the values with one-hot encoding
    #   using `sklearn.preprocessing.OneHotEncoder` (note that its output is by
    #   default sparse, you can use `sparse=False` to generate dense output;
    #   also use `handle_unknown="ignore"` to ignore missing values in test set).
    #
    # - for the rest of the columns, normalize their values so that they
    #   have mean 0 and variance 1; use `sklearn.preprocessing.StandardScaler`.
    #
    # In the output, first there should be all the one-hot categorical features,
    # and then the real-valued features. To process different dataset columns
    # differently, you can use `sklearn.compose.ColumnTransformer`.
    def column_processing(data):
        is_integer_col = []
        not_integer_col = []
        for i in range(data.shape[1]):
            if np.all(np.equal(np.mod(data[:,i], 1), 0)):
                is_integer_col.append(i)
                
            else:
                not_integer_col.append(i)
        integer_encoder = sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
        non_integer_scaler = sklearn.preprocessing.StandardScaler()
        megatron = sklearn.compose.ColumnTransformer(transformers=[("integer_encoder", integer_encoder, is_integer_col), ("non_integer_scaler", non_integer_scaler, not_integer_col)])
        processed_data = megatron.fit(train_data).transform(data)
        return processed_data


    # TODO: To the current features, append polynomial features of order 2.
    # If the input values are [a, b, c, d], you should append
    # [a^2, ab, ac, ad, b^2, bc, bd, c^2, cd, d^2]. You can generate such polynomial
    # features either manually, or you can generate them with
    # `sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)`.
    def add_poly_features(data):
        processed_data = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False).fit_transform(data)
        return processed_data
    # TODO: You can wrap all the feature processing steps into one transformer
    # by using `sklearn.pipeline.Pipeline`. Although not strictly needed, it is
    # usually comfortable.
 
    feature_pipeline = sklearn.pipeline.Pipeline([("transformer", sklearn.preprocessing.FunctionTransformer(column_processing)),("poly", sklearn.preprocessing.FunctionTransformer(add_poly_features))])

    # TODO: Fit the feature processing steps on the training data.
    # Then transform the training data into `train_data` (you can do both these
    # steps using `fit_transform`), and transform testing data to `test_data`.
    
    test_data = feature_pipeline.fit(train_data).transform(test_data)
    train_data = feature_pipeline.fit_transform(train_data)
    return train_data[:5], test_data[:5]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_data, test_data = main(args)
    for dataset in [train_data, test_data]:
        for line in range(min(dataset.shape[0], 5)):
            print(" ".join("{:.4g}".format(dataset[line, column]) for column in range(min(dataset.shape[1], 140))))
