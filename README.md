# 猫狗二分类 + 以图搜图（CNN + MySQL 版本）

完整流程：
1. 按文件名前缀清洗并拆分数据集（0 开头=猫，1 开头=狗）
2. 训练 CNN 分类模型
3. 提取图片特征并存入 **MySQL**
4. 从 MySQL 读取特征，完成以图搜图（只展示 Top5）
# 猫狗二分类 + 以图搜图（CNN 版本）

你提到 MLP 效果差，这里改为 **CNN（ResNet18 迁移学习）**，并补齐完整流程：
1. 按文件名前缀清洗并拆分数据集（0 开头=猫，1 开头=狗）
2. 训练 CNN 分类模型
3. 提取图片特征并存入数据库（SQLite）
4. 从数据库读取特征，完成以图搜图（只展示 Top5）
5. 评估检索准确率（可检查是否达到 98%）

---

## 1) 安装依赖

```bash
pip install -r requirements.txt
```

## 2) 数据清洗与 8:2 拆分

你的原始目录是：
`D:\software\datasets\cat_dog_img\cat_dog_img\img`

执行：

```bash
python prepare_data.py \
  --source-dir "D:\\software\\datasets\\cat_dog_img\\cat_dog_img\\img" \
  --output-dir data \
  --train-ratio 0.8
```

输出目录结构：

```text
data/
  train/
    cat/
    dog/
  val/
    cat/
    dog/
```

---

## 3) 训练 CNN

```bash
python train.py \
  --data-dir data \
  --image-size 224 \
  --batch-size 32 \
  --epochs 20 \
  --lr 3e-4
```

最优模型：`checkpoints/best.pt`

---

## 4) 特征提取并保存到 MySQL

先确保 MySQL 中已有数据库（如 `image_search`）。

```sql
CREATE DATABASE IF NOT EXISTS image_search DEFAULT CHARSET utf8mb4;
```

你的连接参数（按你提供的信息）：
- host: `192.168.2.36`
- user: `root`
- password: `123456`

执行建索引：
训练优化点：
- 迁移学习（ResNet18 预训练权重）
- 数据增强（翻转、旋转、颜色扰动）
- Label smoothing
- AdamW + ReduceLROnPlateau
- Early stopping

最优模型会保存到：`checkpoints/best.pt`

---

## 4) 特征提取并保存到数据库

```bash
python index_features.py \
  --data-dir data \
  --checkpoint checkpoints/best.pt \
  --mysql-host 192.168.2.36 \
  --mysql-port 3306 \
  --mysql-user root \
  --mysql-password 123456 \
  --mysql-database image_search \
  --mysql-table image_features
```

会自动创建表 `image_features`，字段包含：
- `image_path`
- `label`
- `split_name`
- `feature`（LONGBLOB）
  --db-path image_features.db
```

数据库表：`image_features`
- `image_path`
- `label`
- `split`
- `feature`（向量字节）
- `dim`

---

## 5) 以图搜图（Top5）

```bash
python search_image.py \
  --query-image "data/val/cat/0xxx.jpg" \
  --checkpoint checkpoints/best.pt \
  --mysql-host 192.168.2.36 \
  --mysql-port 3306 \
  --mysql-user root \
  --mysql-password 123456 \
  --mysql-database image_search \
  --mysql-table image_features \
  --topk 5
```

脚本会强制最多展示前五个结果。
  --db-path image_features.db \
  --topk 5
```

脚本会从数据库读取全部特征，按余弦相似度排序，只返回前五个最相似图片。

---

## 6) 检索精度评估（目标 98%）

```bash
python eval_retrieval.py \
  --split val \
  --target-top1 0.98 \
  --mysql-host 192.168.2.36 \
  --mysql-port 3306 \
  --mysql-user root \
  --mysql-password 123456 \
  --mysql-database image_search \
  --mysql-table image_features
```

---

## 7) 如果你没接收到（重跑/重推）

可以按下面顺序重新执行一次：

```bash
python index_features.py \
  --data-dir data \
  --checkpoint checkpoints/best.pt \
  --mysql-host 192.168.2.36 \
  --mysql-port 3306 \
  --mysql-user root \
  --mysql-password 123456 \
  --mysql-database image_search \
  --mysql-table image_features

python search_image.py \
  --query-image "data/val/cat/0xxx.jpg" \
  --checkpoint checkpoints/best.pt \
  --mysql-host 192.168.2.36 \
  --mysql-port 3306 \
  --mysql-user root \
  --mysql-password 123456 \
  --mysql-database image_search \
  --mysql-table image_features \
  --topk 5
```

如果数据库连接失败，请先确认：
- MySQL 服务对 `192.168.2.36:3306` 可访问
- `root/123456` 账号有 `image_search` 库的读写权限

## 8) PR 重建说明

如果你在平台上没有收到上一次 PR，可基于当前最新提交重新创建 PR（本次已重新创建）。

## 9) 提交失败重试

如果你看到“没有提交成功”，请直接基于当前分支最新提交重新创建 PR（我这次已经重新创建）。
  --db-path image_features.db \
  --split val \
  --target-top1 0.98
```

> 说明：是否达到 98% 与数据质量、重复样本、类别分布和训练轮数有关。当前脚本会给出 Top1/Top5 指标，并提示是否达到目标。
