# MGS-Net：形态引导分割网络

英文全称为 **Morphology-Guided Segmentation Network**。本仓库提供模型结构、完整损失函数、训练、评估、图片预测和通用数据准备代码。**数据及预训练权重不随仓库提供。** 权重是训练得到的模型参数，需要使用自己的数据训练后保存。

## 安装与合成演示

需要 Python 3.10 或更新版本、PyTorch 2.6 或更新版本，以及 NumPy、Pillow、PyYAML。PyTorch 是模型所用的计算库；已有兼容的 Conda 环境可以直接使用。没有 PyTorch 时，按 [官方安装说明](https://pytorch.org/get-started/locally/)选择与计算机匹配的版本。

在仓库根目录安装本项目和测试依赖：

```bash
python -m pip install -e ".[test]"
python scripts/generate_synthetic_data.py
python scripts/train.py --config configs/synthetic_cpu.yaml --smoke-test
python scripts/evaluate.py --config configs/synthetic_cpu.yaml --checkpoint runs/synthetic/checkpoints/best.pth --device cpu
python scripts/predict.py --config configs/synthetic_cpu.yaml --checkpoint runs/synthetic/checkpoints/best.pth --input generated/synthetic/images/example_000.png --output-dir runs/predictions --device cpu
```

CPU 指普通处理器。冒烟测试指用一个训练批次和一个验证批次检查流程，保存的参数仅来自合成图像。合成示例由简单几何形状直接生成，只用于检查软件是否能运行。演示配置将基础通道数降为4以减少计算；完整模型在 `configs/mgs_net.yaml` 中使用32个基础通道，分支结构和损失定义相同。

再次运行时请为已存在的生成目录、训练目录或预测结果选择新的输出位置。训练和生成数据入口不会自动删除已有内容。

## 使用自己的数据

按照 [数据格式说明](docs/data-format.md)准备灰度图片、外部 JSON 标注及训练、验证和测试清单，在完整配置中修改 `data.root`。清单文件及清单里的路径都相对于这个数据根目录。类别顺序固定为未焊透、未熔合、气孔、夹渣，背景编号为0。

```bash
python scripts/train.py --config configs/mgs_net.yaml
python scripts/evaluate.py --config configs/mgs_net.yaml --checkpoint runs/mgs_net/checkpoints/best.pth --device cpu --batch-size 1
python scripts/predict.py --config configs/mgs_net.yaml --checkpoint runs/mgs_net/checkpoints/best.pth --input data/user/images --output-dir runs/user_predictions --device cpu
```

显卡训练时，单卡设置 `train.device_ids: [0]`，双卡设置 `[0, 1]`。完整配置默认双卡。没有可用显卡时入口会使用普通处理器；可以显式指定 `--device cpu --batch-size 1` 降低内存需求。

数据加载时将灰度值除以255，不自动缩放图像。通常输入高128、宽256；高宽均需为16的倍数。图片预测入口只需要图片与自己训练的参数文件，输出类别编号图、彩色图和叠加图，不读取标注。

## 方法与检查

- 点状、块状、条状三个分支分别提取不同形态的特征。
- 空间增强学习每个位置的增强程度，仅作用于点状和块状分支的初始输入。
- 门控融合学习每个位置三路特征各占多少权重，并将融合信息传回分支。
- 辅助监督在训练时约束各分支识别相应形态；辅助输出是三个独立的二分类结果。
- 损失函数衡量分类、区域重叠、边界和形态约束的误差，详见 [方法说明](docs/method.md)。

IoU（交并比）和 Dice（重叠系数）都衡量预测区域与标注区域的重合程度。评估先跨图片累计每类像素，再计算指标；均值排除背景及没有真实目标像素的类别。没有前景目标时均值为0。

```bash
python -m pytest -q
python scripts/check_release.py
```

所有测试仅使用人工合成内容。代码按 MIT 许可证提供，允许商用、修改和分发，需保留版权及许可声明。

