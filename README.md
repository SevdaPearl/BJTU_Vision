# BJTU_Vision

本项目用于完成《计算机视觉基础》阶段 2 作业中的两个任务：

1. 图像检索任务
2. 文字检测可视化任务

项目当前数据集结构设计：

```
BJTU_Vision/
├── image_retrieval/
│   ├── base/
│   │   ├── BJTU/
│   │   └── util_pic/
│   ├── query/
│   ├── base_statistics.json
│   └── query_statistics.json
│
├── object_detection/
│   └── data/
│       ├── xxx.jpg
│       ├── xxx.json
│       └── ...
```

`object_detection/data` 中的 json 文件为 LabelMe 格式。

## 1. 安装环境

 Python 3.11。

## 2. 检查数据结构

```bash
python scripts/check_data_structure.py
```

如果路径都显示 OK，再运行 notebook。

## 3. 运行顺序

先运行：01_image_retrieval.ipynb

再运行：02_text_detection_labelme.ipynb

## 4. 图像检索任务输出

图像检索任务会读取：

```text
image_retrieval/base/BJTU
image_retrieval/base/util_pic
image_retrieval/query
```

检索阶段不使用地点标签，只使用图片内容特征进行相似度排序。

地点标签仅在评价阶段从文件名前缀提取，用于计算 P@K。

针对12 个地点输出文件：
- `outputs/retrieval_results/retrieval_topk.csv`：每张 query 的 TopK 检索列表。
- `outputs/retrieval_results/precision_by_query.csv`：每张 query 的 P@20、P@40、P@60。
- `outputs/retrieval_results/precision_by_label.csv`：每类 landmark 的平均 P@20、P@40、P@60。
- `outputs/retrieval_results/topk_grids/`：检索效果展示图，不是最终指标图。
- `outputs/pk_curves/`：每类 landmark 一张 P@K 图，共 12 张。

## 5. 文字检测可视化任务输出

读取 LabelMe json 标注文件，输出文件：
- `outputs/detection_results/visual_groups/`：每类 landmark 2 组“检索-检测”可视化结果，共约 24 张。
- `outputs/detection_results/manual_review.csv`：人工核验表。

人工核验：
先打开`outputs/detection_results/manual_review`，
在 `manual_result` 中填写：
   - `OK`：文字框基本正确。
   - `missed_text`：有明显文字没有框出来
   - `false_box`：框到了非文字区域。
   - `unclear`：看不清或不确定。
   - `retrieval_error`：检索错误，不属于本类地点
在 `manual_note` 中写简短说明。

## 6. GitHub 提交说明

提交了代码、README 、演示视频和 notebooks，未提交数据集与大规模运行结果。

`.gitignore` 已经排除了：

```text
image_retrieval/
object_detection/
outputs/cache/
```

如果老师要求展示结果，可以在报告中放关键图片，或把结果截图/演示视频另行提交。
