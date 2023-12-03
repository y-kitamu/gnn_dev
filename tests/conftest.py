"""conftest.py

Author : Yusuke Kitamura
Create Date : 2023-12-03 10:32:26
"""
import os

# testではGPUを使わない
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
