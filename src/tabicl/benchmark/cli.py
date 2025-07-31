#
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
import logging
from pathlib import Path
from typing import List

import click
import pandas as pd
from dr_model_benchmark.tools.openml.utils import get_openml_study
from dr_model_benchmark.tools.openml.utils import get_openml_task
from dr_model_benchmark.tools.openml.utils import get_train_test_sets_of_openml_dataset
from dr_model_benchmark.common.analysis.entities import TestResultV2
from dr_model_benchmark.common.enums import DeviceType
from dr_model_benchmark.common.enums import MetricType
from dr_model_benchmark.common.analysis.enums import Partition
from dr_model_benchmark.common.enums import TargetType
from dr_model_benchmark.common.profile.entities import TimeProfile
from dr_model_benchmark.common.profile.utils import TimeProfiler
from tabicl.benchmark.entities import Dataset, TabICLTestReport
from tabicl.benchmark.models import ModelWrapper
from tabicl.benchmark.evaluations import evaluate_with_cv
from tabicl.benchmark.evaluations import evaluate_on_inference_result
from tabicl import TabICLClassifier


logger = logging.getLogger(__name__)


def infer_classification_target_type(
    classification_train_data: pd.DataFrame, target_name: str
) -> TargetType:
    target_col = classification_train_data[target_name]
    num_of_values = len(target_col.unique())
    return TargetType.BINARY if num_of_values == 2 else TargetType.MULTICLASS


@click.command()
@click.option(
    "--openml_study_id",
    type=int,
    required=True,
    help="OpenML study id (study contains tasks and datasets)",
)
@click.option(
    "--training_metric",
    type=click.Choice([metric_type.name for metric_type in MetricType]),
    required=True,
    help="Metric used for training",
)
@click.option(
    "--evaluation_metrics",
    type=str,
    required=True,
    help="Metric used for evaluation",
)
@click.option(
    "--output_report_path",
    type=str,
    required=True,
    help="OpenML datasets will be downloaded here",
)
@click.option(
    "--device_type",
    type=click.Choice(
        [device_type.name for device_type in DeviceType if device_type != DeviceType.AUTO]
    ),
    required=False,
    default=DeviceType.CUDA,
    help="Device type",
)
def run_cli(
    openml_study_id: int,
    training_metric: str,
    evaluation_metrics: str,
    output_report_path: str,
    device_type: str,
) -> None:
    torch_device_type =DeviceType.from_string(device_type).to_torch_device_type_string()

    openml_study = get_openml_study(openml_study_id)
    logger.info(f"Total {len(openml_study.tasks)} task(s) to test.")
    dataset_test_reports: List[TabICLTestReport] = []
    for openml_task_id in openml_study.tasks:
        openml_task = get_openml_task(openml_task_id)
        openml_dataset = openml_task.get_dataset()
        logger.info(f"Processing task {openml_dataset.name}")
        train_dataframe, test_dataframe = get_train_test_sets_of_openml_dataset(openml_task)
        task_target_name = openml_task.target_name
        target_type = infer_classification_target_type(train_dataframe, task_target_name)
        dataset = Dataset(train_dataframe, test_dataframe, task_target_name)

        # cross validation
        training_metric_type = MetricType.from_string(training_metric)
        model = TabICLClassifier(device=torch_device_type)
        model_wrapper = ModelWrapper(model)
        cv_evaluation_results = evaluate_with_cv(
            model_wrapper,
            dataset,
            5,
            target_type,
            training_metric_type,
        )
        # train
        model = TabICLClassifier(device=torch_device_type)
        model_wrapper = ModelWrapper(model)
        train_fit_time_profile = TimeProfile(Partition.TRAIN.name)
        with TimeProfiler(train_fit_time_profile):
            model_wrapper.fit(dataset)
        # test with external dataset
        holdout_predict_time_profile = TimeProfile(Partition.TEST.name)
        with TimeProfiler(holdout_predict_time_profile):
            prediction_outputs = model_wrapper.inference(dataset)

        evaluation_metric_types = [
            MetricType.from_string(metric) for metric in evaluation_metrics.split(",")
        ]
        holdout_evaluation_results = [
            evaluate_on_inference_result(
                target_type,
                evaluation_metric_type,
                prediction_outputs,
            )
            for evaluation_metric_type in evaluation_metric_types
        ]

        # analysis and report
        dataset_test_reports.append(
            TabICLTestReport(
                openml_dataset.name,
                cv_evaluation_results,
                holdout_evaluation_results,
                [train_fit_time_profile],
                [holdout_predict_time_profile],
            )
        )

    # report
    TestResultV2.to_csv(
        [
            TabICLTestReport.to_test_result(dataset_test_report)
            for dataset_test_report in dataset_test_reports
        ],
        Path(output_report_path),
    )


if __name__ == "__main__":
    run_cli()
