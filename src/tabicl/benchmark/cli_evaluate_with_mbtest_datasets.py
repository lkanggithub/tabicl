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
from dr_model_benchmark.common.analysis.entities import TestResultV2
from dr_model_benchmark.common.enums import DeviceType
from dr_model_benchmark.common.analysis.enums import Partition
from dr_model_benchmark.common.entities import DataRobotMBTestDatasetConfig
from dr_model_benchmark.common.profile.entities import TimeProfile
from dr_model_benchmark.common.profile.utils import TimeProfiler
from tabicl.benchmark.entities import Dataset, TabICLTestReport
from tabicl.benchmark.models import ModelWrapper
from tabicl.benchmark.evaluations import evaluate_with_cv
from tabicl.benchmark.evaluations import evaluate_on_inference_result
from tabicl import TabICLClassifier


logger = logging.getLogger(__name__)


def get_dataset_name(dataset_path: Path) -> str:  # FIXME
    dataset_file_name = dataset_path.name
    return dataset_file_name.split("_train.csv")[0]


@click.command()
@click.option(
    "--datarobot_mbtest_yaml_path",
    type=str,
    required=True,
    help="Path to a DataRobot mbtest yaml file",
)
@click.option(
    "--output_report_path",
    type=str,
    required=True,
    help="Test results will be created here",
)
@click.option(
    "--device_type",
    type=click.Choice([device_type.name for device_type in DeviceType]),
    required=False,
    default=DeviceType.AUTO,
    help="Device type",
)
def run_cli(
    datarobot_mbtest_yaml_path: str,
    output_report_path: str,
    device_type: str,
) -> None:
    datarobot_mbtest_configs = DataRobotMBTestDatasetConfig.load_from_yaml(
        Path(datarobot_mbtest_yaml_path)
    )
    torch_device_type =DeviceType.from_string(device_type).to_torch_device_type_string()

    logger.info(f"Total {len(datarobot_mbtest_configs)} task(s) to test.")
    dataset_test_reports: List[TabICLTestReport] = []
    for mbtest_config in datarobot_mbtest_configs:
        dataset_name = get_dataset_name(Path(mbtest_config.train_dataset_path))
        logger.info(f"Processing task {dataset_name}")

        train_dataframe = pd.read_csv(mbtest_config.train_dataset_path)
        test_dataframe = pd.read_csv(mbtest_config.pred_dataset_path)
        task_target_name = mbtest_config.target
        target_type = mbtest_config.rtype
        dataset = Dataset(train_dataframe, test_dataframe, task_target_name)

        # cross validation
        model = TabICLClassifier(device=torch_device_type)
        model_wrapper = ModelWrapper(model)
        cv_evaluation_results = evaluate_with_cv(
            model_wrapper,
            dataset,
            5,
            target_type,
            mbtest_config.metric,
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

        holdout_evaluation_results = [
            evaluate_on_inference_result(
                target_type,
                mbtest_config.metric,
                prediction_outputs,
            )
        ]

        # analysis and report
        dataset_test_reports.append(
            TabICLTestReport(
                dataset_name,
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
