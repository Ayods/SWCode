# SWCode
该文件为项目介绍文件，主要介绍项目的功能模块。



## 概述

该项目主题为“ 面向门禁人脸识别场景的深度学习模型测试技术 ”。

该项目专为识别模型在多种条件下的性能表现而设计，确保测试的全面性和公平性。项目通过使用标准数据集对选定的人脸识别模型进行测试，并分析其在不同条件下的表现



## 模型

本项目使用deepface模型完成人脸识别功能

[serengil/deepface: A Lightweight Face Recognition and Facial Attribute Analysis (Age, Gender, Emotion and Race) Library for Python](https://github.com/serengil/deepface)



## 数据集

本项目使用公开数据集 **Labeled Faces in the Wild (LFW)**。http://vis-www.cs.umass.edu/lfw/
即项目中的lfw.zip文件

关键步骤包括：

- 选择至少包含两张图片的个体。
- 将数据集限制为最多500名个体（1,000张图片）。



## 功能特点

1. **初始模型评估**：

   测试模型在判断两张图片是否为同一人的基准性能。

2. **基于人口统计学的分析**：

   按以下类别划分数据集并评估模型表现：

   ​	**性别**（男性，女性）。

   ​	**年龄**（0-18岁，19-35岁，36-50岁，51岁以上）。

   ​	**国籍**（如亚洲人、欧洲人、非洲人等）。

3. **数据增强测试**：

   测试模型在图像质量下降情况下的表现：

   ​	降低亮度。

   ​	模糊处理。

4. **全面的评估指标**：

   根据不同数据子集和增强条件评估模型的准确性。



## 测试流程

1. **基准测试**：
   - 对原始数据集进行测试，建立基准准确率。
2. **基于人口统计的测试**：
   - 按性别、年龄组和国籍分类图片。
   - 测试模型在每个子集上的表现，评估其在人口统计学上的差异。
3. **数据增强测试**：
   - 对图像进行增强处理，包括降低亮度和模糊。
   - 测试增强数据集的性能，分析模型在复杂条件下的表现。



## 代码概述

框架使用 Python 实现，并借助 **DeepFace** 库进行人脸识别与分析。主要功能包括：

### 数据准备

```python
# 加载并筛选至少包含两张图片的个体
def loadAndFilter(data_dir, min_images=2, max_people=500):
```

### 分类功能

- **性别分类**：

```python
def classifyByGender(image_paths):
```

- **年龄分类**：

```python
def classifyByAge(image_paths):
```

- **国籍分类**：

```python
def classifyByNationality(image_paths):
```

### 图像增强

```python
# 生成降低亮度和模糊处理的增强图片
def augmentImages(image_paths, output_dir):
```

### 测试功能

- **基准和分类测试**：

```python
# 测试原始或分类数据集的模型准确性
def test1(image_paths, model_name="VGG-Face"):
```

- **增强数据测试**：

```python
# 测试增强数据集的模型准确性
def test2(enhanced_images):
```

### 主脚本

```python
if __name__ == "__main__":
    # 数据准备和筛选
    data_dir = "lfw"
    qualified_images = loadAndFilter(data_dir, max_people=500)

    # 基准测试
    print("测试原始数据集的准确率...")
    test1(qualified_images)

    # 性别分类测试
    male_images, female_images = classifyByGender(qualified_images)
    test1(male_images)
    test1(female_images)

    # 年龄分类测试
    age_groups = classifyByAge(qualified_images)
    for age_group, images in age_groups.items():
        test1(images)

    # 国籍分类测试
    nationality_groups = classifyByNationality(qualified_images)
    for nationality, images in nationality_groups.items():
        test1(images)

    # 增强数据集测试
    augmented_dir = "augmented_lfw_images"
    dark_images, blur_images = augmentImages(qualified_images, augmented_dir)
    test2(dark_images)
    test2(blur_images)
```



## 测试结果

1. 在原始数据集上评估了基准准确率。
2. 性别、年龄组和国籍子集的测试结果显示了性能差异。
3. 数据增强测试表明，在低亮度和模糊条件下准确率有所下降。

![image-20241229214518599](https://github.com/Ayods/SWCode/blob/main/result.png)



## 依赖项

- **Python**：3.7+
- **DeepFace**：最新版本
- **OpenCV**：用于图像处理
- **NumPy**：用于数值计算

