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
from dr_model_benchmark.datarobot.predictions import PredictionOutputs

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
        return self

    def inference(self, dataset: Dataset) -> PredictionOutputs:
        inference_input_data = dataset.get_test_data_x()
        prediction_values = self.pipeline.predict(inference_input_data)
        prediction_proba_values = (
            self.pipeline.predict_proba(inference_input_data)
            if not self.is_regressor
            else None
        )
        class_labels = self.model.classes_ if not self.is_regressor else None

        return PredictionOutputs(
            actual_values=dataset.get_test_data_y(),
            prediction_values=prediction_values,
            prediction_proba_values=prediction_proba_values,
            class_labels=class_labels,
        )
