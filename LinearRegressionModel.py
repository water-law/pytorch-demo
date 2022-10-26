import numpy as np
import torch
import torch.nn as nn


class LinearRegressionModel(nn.Module):
    # 输入维度  输出维度
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 输入输出数据 x_train --> y_train
    x_values = [i for i in range(11)]
    x_train = np.array(x_values, dtype=np.float32)
    # -1 表示任意行， 1 表示 1 列
    x_train = x_train.reshape(-1, 1)
    print(x_train)
    y_values = [2 * i + 1 for i in range(11)]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)
    print(y_train)

    input_dim = 1
    output_dim = 1
    # 使用 GPU 训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LinearRegressionModel(input_dim, output_dim)
    # 传到 GPU
    model.to(device)
    print(model)
    # 指定参数和损失函数
    # 学习次数
    epochs = 1000
    # 学习率
    learning_rate = 0.01
    # 优化器采用 SGD: 随机梯度下降（stochastic gradient descent）
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # 使用平均平方差来计算损失
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch += 1
        # 转换为张量 tensor
        inputs = torch.from_numpy(x_train).to(device)
        labels = torch.from_numpy(y_train).to(device)
        # 梯度清零，默认会累加
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播， 求导数
        loss.backward()
        # 更新权重参数, y = w * x + b (权重参数为 w 和 b)
        optimizer.step()
        if epoch % 50 == 0:
            print("epoch: {}, loss: {}".format(epoch, loss.item()))

    # 测试模型预测结果
    predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
    print(predicted)
    # 模型的保存与读取
    torch.save(model.state_dict(), 'model.pkl')
    # 读取
    # model.load_state_dict(torch.load('model.pkl'))