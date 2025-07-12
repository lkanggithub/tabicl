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
from typing import Callable
from typing import List

from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from foundation_model_compare.common.enums import MetricType
from foundation_model_compare.common.enums import TargetType
from foundation_model_compare.tabpfn.entities import CVEvaluationResult
from foundation_model_compare.tabpfn.entities import Dataset
from foundation_model_compare.tabpfn.entities import EvaluationResult
from foundation_model_compare.tabpfn.entities import InferenceResults

from tabicl.benchmark.models import ModelWrapper


def get_sklearn_scorer(
    target_type: TargetType,
    metric_type: MetricType,
) -> Callable:
    is_multiclass = target_type == TargetType.MULTICLASS
    scorer_args = {}
    if target_type.is_classification():
        scorer_args.update({"response_method": "predict_proba"})
    if is_multiclass:
        scorer_args.update({"multi_class": "ovr"})

    return make_scorer(get_sklearn_score_func(metric_type), **scorer_args)


def get_sklearn_score_func(metric_type: MetricType) -> Callable:
    metric_handler = {
        MetricType.AUC: roc_auc_score,
        MetricType.RMSE: root_mean_squared_error,
    }
    return metric_handler[metric_type]


def evaluate_on_inference_result(
    target_type: TargetType,
    metric_type: MetricType,
    inference_result: InferenceResults,
) -> EvaluationResult:
    if target_type == TargetType.BINARY:
        prediction_actual = inference_result.prediction_probabilities[:, 1]
    elif target_type == TargetType.MULTICLASS:
        prediction_actual = inference_result.prediction_probabilities
    else:
        prediction_actual = inference_result.predictions

    score_func_extra_args = {}
    if target_type == TargetType.MULTICLASS:
        score_func_extra_args.update({"multi_class": "ovr"})
    score_func = get_sklearn_score_func(metric_type)
    score = score_func(
        inference_result.prediction_true,
        prediction_actual,
        **score_func_extra_args,
    )

    return EvaluationResult(metric_type, score)


def evaluate_with_cv(
    model_wrapper: ModelWrapper,
    dataset: Dataset,
    num_of_folds: int,
    target_type: TargetType,
    metric_type: MetricType,
) -> List[CVEvaluationResult]:
    cv = KFold(n_splits=num_of_folds, shuffle=True, random_state=1234)
    cv_scores = cross_val_score(
        estimator=model_wrapper.pipeline,
        # estimator=model_wrapper.model,
        X=dataset.get_train_data_x(),
        y=dataset.get_train_data_y(),
        cv=cv,
        scoring=get_sklearn_scorer(target_type, metric_type),
    )

    return [
        CVEvaluationResult(EvaluationResult(metric_type, cv_score), idx)
        for idx, cv_score in enumerate(cv_scores)
    ]
