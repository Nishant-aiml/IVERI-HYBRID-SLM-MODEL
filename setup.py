# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""IVERI CORE — Package setup for local development installation."""

from setuptools import find_packages, setup

setup(
    name="iveri-core",
    version="0.1.0",
    packages=find_packages(
        include=[
            "configs*",
            "core*",
            "model*",
            "data*",
            "training*",
            "evaluation*",
            "baselines*",
            "utils*",
            "scripts*",
        ]
    ),
    python_requires=">=3.10",
)
