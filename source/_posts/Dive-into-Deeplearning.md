---
title: Dive into Deeplearning
date: 2023-09-14 23:09:48
categories:
- work
tags:
- AI
- Deeplearning
- 课程笔记
---

作为一切的开始，重新整理一份笔记，便于后期快速复习
<!-- more -->

参考：
- [《动手学深度学习》](https://zh.d2l.ai/index.html)
- [跟李沐学AI](https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497&ctype=0)

![全书结构](image.png)

## Intro

### 机器学习的关键组件：
1. 用来学习的数据：data
2. 转换数据的模型：model
3. 用来量化模型有效性的目标函数：objective function（lose function）
4. 调整模型参数以优化目标函数得到的值的算法：algorithm

#### 数据
用于机器学习的数据一般可分为训练集（train）、测试集（validate）和验证集（test）
1. 训练集用于训练即反向传播调整模型参数
2. 测试集用于调整模型超参数
3. 验证集用于评估模型最终性能

每个数据集由一个个样本（sample，也称数据点或数据实例）组成，大多时候这些样本都遵循独立同分布，每个样本由一组称为特征（features）的属性组成，特征数量称为该样本数据的维数（dimensionality）。机器学习模型会根据这些属性进行预测，在监督学习中，预测结果是一个特殊属性，称为标签（label）

#### 模型
这是深度学习与经典机器学习的主要区别点：深度学习的模型更复杂，数据转换层数更多

#### 目标函数
即误差函数，一般用来度量预测值与真实值之间的误差

#### 优化算法
用于搜索模型的最佳参数，从而最小化目标函数。一般采用梯度下降

### 机器学习问题分类

#### 监督学习

