import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv('encode.csv')

# 映射岗位ID到从0到19的整数标签
id_mapping = {id: idx for idx, id in enumerate(data['岗位ID'].unique())}
data['岗位ID'] = data['岗位ID'].map(id_mapping)

# 标签编码器
label_encoder = LabelEncoder()

# 对分类特征进行标签编码
categorical_features = ["职位名", "学校", "专业", "项目名称", "学校层次", "期望职位", "组织名", "工作职位", "公司名称",
                        "工作职位1", "公司名称1", "专业1", "学校1", "学校层次1", "职位名1", "组织名1", "项目名称1"]
for feature in categorical_features:
    data[feature] = label_encoder.fit_transform(data[feature])

# 标准化数值特征
numeric_features = ["工作经历数量", "社会经历数量", "项目数量", "技能数量", "荣誉数量"]
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# 划分特征和标签
X = data.drop(columns=["岗位ID"])
y = data["岗位ID"]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 将数据转换为PyTorch张量并移至CPU
X_train = torch.tensor(X_train.values, dtype=torch.float32).cpu()
y_train = torch.tensor(y_train.values, dtype=torch.long).cpu()
X_test = torch.tensor(X_test.values, dtype=torch.float32).cpu()
y_test = torch.tensor(y_test.values, dtype=torch.long).cpu()

# 定义TabNet模型
clf = TabNetClassifier(
    n_d=32,  # Dimension of the embeddings for categorical features
    n_a=32,  # Dimension of the embeddings for numerical features
    n_steps=4,
    gamma=1.5,
    cat_idxs=[X.columns.get_loc(feature) for feature in categorical_features],
    cat_dims=[len(data[feature].unique()) for feature in categorical_features],
    cat_emb_dim=1,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size": 10, "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax',  # Choose 'sparsemax' or 'entmax' for categorical features
    verbose=1,
    device_name="cpu"  # 将模型设置为在CPU上运行
)

# 训练模型
clf.fit(
    X_train=X_train.numpy(),
    y_train=y_train.numpy(),
    eval_set=[(X_test.numpy(), y_test.numpy())],
    max_epochs=50,
    patience=10,  # Early stopping patience
    batch_size=32,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# 在测试集上评估模型
test_preds = clf.predict_proba(X_test.numpy())
test_preds_class = np.argmax(test_preds, axis=1)
accuracy = (test_preds_class == y_test.numpy()).mean()
print(f"Test Accuracy: {accuracy * 100:.2f}%")
