#!/usr/bin/env python3 -*- coding: utf-8 -*-

import os

from setuptools import find_packages, setup


def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("webscale_rl", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]


setup(
    name="webscale_rl",
    version=get_version(),
    packages=find_packages(
        exclude=["tests", "tests.*", "examples", "examples.*", "docs", "docs.*"]
    ),
)