import numpy as np

# 定义tanh函数
def tanh(x):
    return np.tanh(x)

# 定义tanh函数的导数
def tanh_deriv(x):
    return 1.0 - np.tanh(x) * np.tanh(x)

#定义logistic函数
def logistic(x):
    return 1/(1 + np.exp(-x))

#定义logistic的导数
def logistic_deriv(x):
    return logistic(x)*(1-logistic(x))

class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        '''

        :param layers: 接受一个list，每个元素为每层神经元的个数,至少为两层
        :param activation:算法类型，默认tanh,可选logistic
        '''

        if activation == 'logistic':
            self.activation = logistic
            self.activation_derive = logistic_deriv
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derive = tanh_deriv
        #初始化权重
        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)

    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        '''

        :param X:
        :param y:输出标记 0或者1
        :param learning_rate:学习率
        :param epochs:设定神经网络循环次数 (神经网络停止条件) 1.权重低于某个阈值 2.预测的错误率低于某个阈值 3.设置训练次数
        :return:
        '''

        X = np.atleast_2d(X)
        temp = np.ones(X.shape[0], X.shape[1] + 1)
        temp[:, 0:-1] = X           #初始化bias
        X = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(X.shape[0]) #随机抽取实例
            a = [X[i]]
            #计算下一节点的值
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            #算出error
            error = y[i] - a[-1]
            deltas = [error * self.activation_derive(a[-1])]
            #开始反向更新
