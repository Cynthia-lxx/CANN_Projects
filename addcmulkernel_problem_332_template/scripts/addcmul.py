#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Addcmul算子实现
公式: y = input_data + x1 * x2 * value
参考: torch.addcmul
"""
import numpy as np


def impl(input_data, x1, x2, value):
    """
    Addcmul算子实现
    
    参数:
    input_data: 第一个输入张量
    x1: 第二个输入张量
    x2: 第三个输入张量
    value: 标量值（形状为[1]）
    
    返回:
    y: 计算结果张量
    
    功能:
    1. 支持 NumPy 广播语义
    2. 支持多种数据类型: float16, float32, int8, int32
    3. 处理非对齐内存场景
    4. 支持高维张量计算
    """
    # value 是形状为 [1] 的数组，取其标量值
    v = value.flatten()[0] if value.size == 1 else value
    
    # 计算公式: y = input_data + x1 * x2 * value
    result = input_data + x1 * x2 * v
    
    return result.astype(input_data.dtype)
