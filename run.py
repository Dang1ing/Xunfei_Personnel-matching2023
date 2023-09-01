import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 假设数据存储在名为 'data.csv' 的CSV文件中，读取数据
data = pd.read_csv('data.csv')

# 分离特征和目标
X = data.drop(columns=['岗位ID'])  # 特征矩阵
y = data['岗位ID']  # 目标（岗位ID）

# 划分数据集为训练集和测试集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 定义类别型特征和数值型特征
categorical_features = ['工作职位', '公司名称',
                        '专业', '学校', '学校层次', '职位名', '组织名', '项目名称']
numeric_features = ['工作经历数量', '社会经历数量', '项目数量', '技能数量', '荣誉数量']

# 创建预处理管道
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())  # 数值型特征标准化
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # 类别型特征独热编码
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 在训练集上拟合预处理管道
X_train_preprocessed = preprocessor.fit_transform(X_train)

# 在验证集上应用预处理管道
X_val_preprocessed = preprocessor.transform(X_val)

# 输出预处理后的特征矩阵和目标
print("预处理后的特征矩阵：")
print(X_train_preprocessed)
print("训练集目标：")
print(y_train)
