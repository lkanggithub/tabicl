# Copyright 2025 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
#
# DataRobot, Inc. Confidential.
#
# This is unpublished proprietary source code of DataRobot, Inc.
# and its affiliates.
#
# The copyright notice above does not evidence any actual or intended
# publication of such source code.
from typing import Optional

from skrub import TableVectorizer
from sklearn.pipeline import make_pipeline

from foundation_model_compare.tabpfn.entities import InferenceResults
from tabicl.benchmark.entities import Dataset

from tabicl import TabICLClassifier


class ModelWrapper:
    def __init__(
        self,
        model: TabICLClassifier,
        inference_only: Optional[bool] = False,
    ):
        self.model = model
        self.is_regressor = not isinstance(self.model, TabICLClassifier)
        self.inference_only = inference_only
        self.pipeline = make_pipeline(
            TableVectorizer(),  # Automatically handles various data types
            self.model,
        )

    def fit(self, dataset: Dataset) -> "ModelWrapper":
        if self.inference_only:
            return self

        self.pipeline.fit(dataset.get_train_data_x(), dataset.get_train_data_y())

        # self.model.fit(dataset.get_train_data_x(), dataset.get_train_data_y())
        return self

    def inference(
        self,
        dataset: Dataset,
        include_true_prediction: Optional[bool] = False,
    ) -> InferenceResults:
        data = dataset.get_test_data_x()
        # result = (
        #     InferenceResults(self.model.predict(data))
        #     if self.is_regressor
        #     else InferenceResults(self.model.predict(data), self.model.predict_proba(data))
        # )

        result = (
            InferenceResults(self.pipeline.predict(data))
            if self.is_regressor
            else InferenceResults(self.pipeline.predict(data), self.pipeline.predict_proba(data))
        )

        if include_true_prediction:
            result.prediction_true = dataset.get_test_data_y()

        return result
