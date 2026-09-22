# -*- coding: utf-8 -*-
"""pytest 选项：--run-live（默认关闭，live ping 专用）。"""


def pytest_addoption(parser):
    parser.addoption("--run-live", action="store_true", default=False,
                     help="运行 live ping（默认跳过；1 call 上限，只打便宜模型）")
