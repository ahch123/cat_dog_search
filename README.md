# 猫狗二分类 + 以图搜图（CNN + MySQL 版本）

完整流程：
1. 按文件名前缀清洗并拆分数据集（0 开头=猫，1 开头=狗）
2. 训练 CNN 分类模型
3. 提取图片特征并存入 **MySQL**
4. 从 MySQL 读取特征，完成以图搜图（只展示 Top5）
5. 评估检索准确率（可检查是否达到 98%）

---

## 1) 安装依赖

```bash
pip install -r requirements.txt
```

## 2) 数据清洗与 8:2 拆分

```bash
python prepare_data.py \
  --source-dir "D:\\software\\datasets\\cat_dog_img\\cat_dog_img\\img" \
  --output-dir data \
  --train-ratio 0.8
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
