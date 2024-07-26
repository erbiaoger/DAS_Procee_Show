  # 数据处理及可视化-        DAS 数据处理及可视化软件   -    使用说明    

| 版本|            1.0|
| --------- | ---- |
| 文档状态: | 编辑 |
|作者:      |      张志宇        |
|负责人:    |        张志宇    |
|建日期:            |2023年8月01日    |
|更新日期:     |       2023年8月10日 |                                                                            

修订历史

| **日期**  | **版本** | **修改者** | **描述**       |
| --------- | -------- | ---------- | -------------- |
| 2023-8-01 | 0.8      | 张志宇     | 完善了产品简介 |
| 2023-8-10 | 1.0      | 张志宇     | 完善了使用说明 |
|           |          |            |                |
|           |          |            |                |
|           |          |            |                |
|           |          |            |                |
|           |          |            |                |
|           |          |            |                |

 

 

目录


- [数据处理及可视化-        DAS 数据处理及可视化软件   -    使用说明](#数据处理及可视化---------das-数据处理及可视化软件--------使用说明)
- [1  简介](#1--简介)
  - [1.1  编写目的](#11--编写目的)
  - [1.2  使用对象](#12--使用对象)
  - [1.3  产品范围](#13--产品范围)
- [2  使用说明](#2--使用说明)
  - [2.1   安装](#21---安装)
  - [2.2   启动dasQt](#22---启动dasqt)
  - [2.3   数据导入](#23---数据导入)
  - [2.4   数据可视化](#24---数据可视化)
    - [2.4.1   能量图](#241---能量图)
    - [2.4.2   震动记录](#242---震动记录)
    - [2.4.3   开始动图](#243---开始动图)
    - [2.4.4   调节动图速度](#244---调节动图速度)
    - [2.4.5   调节图片颜色尺度](#245---调节图片颜色尺度)
    - [2.4.6   调节图片时间](#246---调节图片时间)
    - [2.4.7   局部放大](#247---局部放大)
  - [2.5   数据处理](#25---数据处理)
    - [2.5.1   滤波](#251---滤波)
    - [2.5.2   坏道切除](#252---坏道切除)
  - [2.6   Pick Points](#26---pick-points)
    - [2.6.1   打开Pick Points模块](#261---打开pick-points模块)
    - [2.6.2   导入数据集](#262---导入数据集)
    - [2.6.3   数据可视化](#263---数据可视化)
    - [2.6.4   调节图片颜色尺度](#264---调节图片颜色尺度)
    - [2.6.5   绘点](#265---绘点)
    - [2.6.6   下一个数据](#266---下一个数据)

 



 

# 1  简介

分布式光纤声波传感（Distributed Fiber Optic Acoustic Sensing）是一种先进的技术，利用光纤作为传感元件来实现对声波信号的实时监测和定位。它通过利用光纤的敏感性，以及光信号与声波相互作用的原理，实现对声波事件的分布式监测和分析。

这项技术的核心是光纤的布里渊散射效应。当声波在光纤附近传播时，它会导致光纤中的光子发生布里渊散射，这会导致光信号的频率发生变化。通过分析这种频率变化，可以确定声波的产生、传播方向、幅度等信息。

分布式光纤声波传感的工作原理如下：

1. **激光光源：** 一个激光器通过光纤发送激光信号，作为传感的基础。
2. **光纤传感区域：** 光纤被布置在需要监测的区域，声波在这个区域内产生并传播。
3. **声波与光相互作用：** 声波的存在导致光纤中的光子发生布里渊散射，这会引起光信号的频率偏移。
4. **频率分析：** 通过对返回的光信号进行频率分析，可以确定声波的性质，如频率、强度和传播方向。
5. **数据处理与可视化：** 收集到的数据可以通过算法进行处理，以生成声波事件的分布图、频谱图等可视化结果。

这种技术在许多领域都有应用，包括但不限于：

·    **地震监测：** 可以用于监测地壳的地震活动，提供早期警报和地震事件分析。

·    **管道泄漏检测：** 可以检测沿着光纤布置的管道上的泄漏事件，有助于保护环境和资源。

·    **智能交通系统：** 可以用于交通流量监测、车辆定位等，改善交通管理。

·    **结构健康监测：** 可以监测建筑物、桥梁等基础设施的结构变化，提前发现可能的问题。

·    **军事和安全应用：** 可以用于监测敌方活动、侦测爆炸声等。

## 1.1  编写目的

本文档为使用说明文档，为产品的使用与维护提供信息基础。

## 1.2  使用对象

本文档的使用对象主要为产品测试与使用人员。

## 1.3  产品范围

本软件通过精密的数据处理和分析，提供了一系列强大的功能，包括：

1. **数据管理与处理：** DAS 数据处理及可视化软件能够高效管理和处理来自DAS采集的大量数据，确保数据的准确性和完整性。
2. **数据可视化：** 软件提供直观的数据可视化工具，将复杂的探测数据转化为易于理解的图像和图表，助力用户更深入地探索地下信息。
3. **数据挑选：**为了更好地聚焦分析，DAS 数据处理及可视化软件提供数据挑选功能，使用户能够选择感兴趣的数据子集进行进一步的处理和研究。通过设定特定的参数和条件，用户可以从大量数据中筛选出与其研究目标相关的数据，从而提高效率并准确地针对性地分析。

 

# 2  使用说明



## 2.1   安装

1. 安装依赖库

使用 **pip install -r requirements.txt**

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image002.jpg)

2. 安装dasQt 

使用 **pip install -e .**

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image004.jpg)

## 2.2   启动dasQt 

在命令行中输入 dasQt 即可启动程序

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image006.jpg)

![图形用户界面, 文本, 应用程序  描述已自动生成](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image008.jpg)

## 2.3   数据导入

鼠标悬停在**导入数据**按钮上可查看可导入数据类型

 

![图形用户界面, 文本, 应用程序  描述已自动生成](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-252023-11-25clip_image010.jpg)

点击示例数据，打开

![img](Users/zhiyuzhang/Library/Group Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/clip_image012.jpg)

## 2.4   数据可视化

### 2.4.1   能量图

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image014.jpg)

### 2.4.2   震动记录

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image016.jpg)

### 2.4.3   开始动图

点击 start Animation 按钮，数据开始以动图形式呈现。

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image018.jpg)

### 2.4.4   调节动图速度

拉动Speed 滑块

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image020.jpg)

### 2.4.5   调节图片颜色尺度

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image022.jpg)

### 2.4.6   调节图片时间

点击Prev time，表示前一个时刻；点击Next time，表示后一个时刻。

### 2.4.7   局部放大

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image024.jpg)

## 2.5   数据处理

### 2.5.1   滤波

进行2～30Hz带通滤波，可以明显看出差异。

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image026.jpg)

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image028.jpg)

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image030.jpg)

### 2.5.2   坏道切除

。

 

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image032.jpg)

## 2.6   Pick Points

拾取DAS数据中的车辆信号。

### 2.6.1   打开Pick Points模块

点击Pick Points

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image034.jpg)

### 2.6.2   导入数据集

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image036.jpg)

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image038.jpg)

### 2.6.3   数据可视化

点击数据名称

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image040.jpg)

### 2.6.4   调节图片颜色尺度

拉动滑块，调节颜色范围。

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image042.jpg)

### 2.6.5   绘点

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image044.jpg)

### 2.6.6   下一个数据

点击Next

![img](https://raw.githubusercontent.com/erbiaoger/PicGo/main/2023-11-25clip_image046.jpg)

 