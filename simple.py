import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences  # 使用Keras的padding函数

# 加载数据集
data = pd.read_csv("data.csv")  # 替换成您的数据文件路径

# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 创建岗位ID到整数类别的映射字典
unique_positions = train_data['岗位ID'].unique()
position_mapping = {pos: idx for idx, pos in enumerate(unique_positions)}
train_data['岗位ID'] = train_data['岗位ID'].map(position_mapping)
test_data['岗位ID'] = test_data['岗位ID'].map(position_mapping)

# 数据预处理
scaler = StandardScaler()
numeric_features = ["工作经历数量", "社会经历数量", "项目数量", "技能数量", "荣誉数量"]

# 处理多取值的分类特征
embedding_features = ["职位名", "学校", "专业", "项目名称",
                      "学校层次", "期望职位", "组织名", "工作职位", "公司名称",
                      "工作职位1", "公司名称1", "专业1", "学校1", "学校层次1", "职位名1", "组织名1", "项目名称1"]
embedding_dims = [10, 10, 10, 10, 10, 10, 10, 10, 10]

# 构建嵌入层
embedding_layers = nn.ModuleList([
    nn.Embedding(len(data[feature].unique()) + 1, dim) for feature, dim in zip(embedding_features, embedding_dims)
])

# 数据预处理：标准化数值特征
train_data[numeric_features] = scaler.fit_transform(
    train_data[numeric_features])
test_data[numeric_features] = scaler.transform(test_data[numeric_features])

# 构建深度模型


class MultiClassMatchingModel(nn.Module):
    def __init__(self, embedding_layers, input_dim, hidden_dim, num_classes):
        super(MultiClassMatchingModel, self).__init__()
        self.embedding_layers = embedding_layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # 处理嵌入特征
        embedded = [emb_layer(x[:, i].long())
                    for i, emb_layer in enumerate(self.embedding_layers)]
        embedded = torch.cat(embedded, dim=1)

        # 处理数值特征
        numeric = x[:, len(embedding_features):]

        # 合并嵌入特征和数值特征
        x = torch.cat([embedded, numeric], dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 定义模型参数
input_dim = sum(embedding_dims) + len(numeric_features)  # 输入特征维度
hidden_dim = 128  # 隐藏层维度
num_classes = 20  # 岗位ID的类别数

# 创建模型实例
model = MultiClassMatchingModel(
    embedding_layers, input_dim, hidden_dim, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 准备训练数据


def prepare_input_data(data):
    cat_features = []
    for feature in embedding_features:
        # 使用Keras的pad_sequences来填充或截断序列至相同长度
        sequences = data[feature].apply(
            lambda x: [int(value) for value in x[1:-1].split()]).tolist()
        padded_sequences = pad_sequences(
            sequences, maxlen=max_sequence_length, padding='post', truncating='post')
        cat_features.append(torch.LongTensor(padded_sequences))

    # 将数值特征转换为 FloatTensor
    num_features = torch.FloatTensor(data[numeric_features].values)

    # 合并所有特征
    input_data = torch.cat(cat_features + [num_features], dim=1)
    return input_data


# 确定最大序列长度
max_sequence_length = max(train_data[embedding_features].apply(lambda x: len(x[1:-1].split())).max(),
                          test_data[embedding_features].apply(lambda x: len(x[1:-1].split())).max())


# 在测试集上评估模型
model.eval()
test_x = prepare_input_data(test_data)
test_y = torch.LongTensor(test_data['岗位ID'].values)

with torch.no_grad():
    test_outputs = model(test_x)
    predicted_labels = torch.argmax(
        test_outputs, dim=1).numpy()  # 使用最大概率的类别作为预测结果

# 计算准确率
accuracy = accuracy_score(test_y.numpy(), predicted_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
