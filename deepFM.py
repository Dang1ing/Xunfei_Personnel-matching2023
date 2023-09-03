import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
# for feature in categorical_features:
#     data[feature] = label_encoder.fit_transform(data[feature])

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

# 将数据转换为PyTorch张量
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.long)

# 确定输入维度
input_dim = X_train.shape[1]

# 定义DeepFM模型


class DeepFM(nn.Module):
    def __init__(self, num_features, embed_dim, hidden_dim):
        super(DeepFM, self).__init__()
        # Embedding层
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_features, embed_dim) for _ in range(X_train.shape[1])])
        self.linear = nn.Linear(embed_dim, 1)
        self.fm = nn.Parameter(torch.randn(embed_dim))
        self.hidden_layers = nn.Sequential(
            nn.Linear(X_train.shape[1], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(
            embed_dim + hidden_dim, len(data["岗位ID"].unique()))

    def forward(self, x):
        # Embedding层
        embeddings = [emb(x[:, i].long())
                      for i, emb in enumerate(self.embeddings)]
        embeddings = torch.stack(embeddings, dim=1)

        # FM部分
        fm_term_1 = torch.sum(embeddings, dim=1)
        fm_term_2 = torch.sum(embeddings**2, dim=1)
        fm_output = 0.5 * (fm_term_1**2 - fm_term_2)

        # DNN部分
        dnn_output = self.hidden_layers(x)

        # 结合FM和DNN部分
        final_output = torch.cat([fm_output, dnn_output], dim=1)
        final_output = self.output_layer(final_output)

        return final_output


# # 初始化模型
# num_features = len(data[categorical_features].stack().unique())
# embed_dim = 10
# hidden_dim = 64
# model = DeepFM(num_features, embed_dim, hidden_dim)
# 初始化模型
embed_dim = 10
hidden_dim = 64
model = DeepFM(input_dim, embed_dim, hidden_dim)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 50
batch_size = 64

for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# 在测试集上评估模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
