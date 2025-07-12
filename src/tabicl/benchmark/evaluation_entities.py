from dataclasses import dataclass
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from foundation_model_compare.analysis.entities import TestResult
from foundation_model_compare.common.enums import MetricType
from foundation_model_compare.tabpfn.profiler import TimeProfile


class Dataset:
    def __init__(
        self,
        train_data_path: str,
        test_data_path: str,
        target_name: str,
    ):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.target_name = target_name
        self.train_data = self.get_train_data()
        self.test_data = self.get_test_data()

    def get_train_data(self) -> pd.DataFrame:
        return pd.read_csv(self.train_data_path)

    def get_test_data(self) -> pd.DataFrame:
        return pd.read_csv(self.test_data_path)

    def get_train_data_x(self) -> np.ndarray:
        columns = list(self.train_data.columns)
        columns.remove(self.target_name)
        return self.train_data[columns].values

    def get_train_data_y(self) -> np.ndarray:
        return self.train_data[self.target_name].values

    def get_test_data_x(self) -> np.ndarray:
        columns = list(self.test_data.columns)
        columns.remove(self.target_name)
        return self.test_data[columns].values

    def get_test_data_y(self) -> np.ndarray:
        return self.test_data[self.target_name].values


@dataclass
class InferenceResults:
    predictions: np.ndarray
    prediction_probabilities: np.ndarray = None
    prediction_true: np.ndarray = None


@dataclass
class EvaluationResult:
    metric_type: MetricType
    score: Optional[float] = None


@dataclass
class CVEvaluationResult:
    evaluation_result: EvaluationResult
    cv_fold: int


@dataclass
class TabPFNTestReport:
    name: str
    metric_type: MetricType
    cv_evaluation_results: List[CVEvaluationResult]
    test_set_evaluation_results: List[EvaluationResult]
    train_time_profile: List[TimeProfile]
    test_set_predict_time_profile: List[TimeProfile]

    def median_cv_score(self) -> np.floating:
        return np.median(
            [eval_result.evaluation_result.score for eval_result in self.cv_evaluation_results]
        )

    def median_test_set_score(self) -> np.floating:
        return np.median([eval_result.score for eval_result in self.test_set_evaluation_results])

    def median_train_time(self) -> np.floating:
        return np.median(
            [time_profile.time_ellipse.to_float() for time_profile in self.train_time_profile]
        )

    def median_test_set_predict_time(self) -> np.floating:
        return np.median(
            [
                time_profile.time_ellipse.to_float()
                for time_profile in self.test_set_predict_time_profile
            ]
        )

    def to_test_result(self) -> TestResult:
        return TestResult(
            dataset_name=self.name,
            metric_type=self.metric_type,
            cv_score=self.median_cv_score().item(),
            test_score=self.median_test_set_score().item(),
            fit_clock_time=self.median_train_time().item(),
            predict_clock_time=self.median_test_set_predict_time().item(),
        )
