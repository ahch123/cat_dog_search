# 猫狗二分类（全连接神经网络）

这个项目提供一个**从零开始**的猫狗二分类示例，使用全连接神经网络（MLP）完成训练与评估。

## 1. 环境准备

建议使用 Python 3.9+。

```bash
pip install -r requirements.txt
```

## 2. 数据集准备

如果你的图片都放在一个目录中（例如 `D:\software\datasets\cat_dog_img\cat_dog_img\img`），并且**文件名以 0 开头的是猫、以 1 开头的是狗**，可以使用脚本按 8:2 自动拆分训练集与验证集：

```bash
python prepare_data.py \
  --source-dir "D:\\software\\datasets\\cat_dog_img\\cat_dog_img\\img" \
  --output-dir data \
  --train-ratio 0.8
```

运行后会得到如下目录结构（使用 `torchvision.datasets.ImageFolder`）：
请将数据集整理成如下目录结构（使用 `torchvision.datasets.ImageFolder`）：

```
workspace/xiaoy/
  data/
    train/
      cat/
        *.jpg
      dog/
        *.jpg
    val/
      cat/
        *.jpg
      dog/
        *.jpg
```

> 你可以使用 Kaggle 的 Cats vs Dogs 或其他猫狗图片数据集，只要整理成上述结构即可。

## 3. 训练

```bash
python train.py \
  --data-dir data \
  --image-size 128 \
  --batch-size 64 \
  --epochs 10 \
  --lr 1e-3
```

训练完成后会在 `checkpoints/` 目录下保存最优模型（`best.pt`）。

## 4. 评估

训练过程中会在每个 epoch 结束后输出验证集准确率。

## 5. 说明

这是一个**纯全连接网络**示例：输入图像先被展平（flatten），再进入 MLP。该方案易于理解但对图像任务而言性能不如 CNN，适合入门和教学演示。
