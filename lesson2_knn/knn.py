import csv
import random
import math
import operator

'''
1.加载样本
2.得到邻居
3.进行投票分类
'''


#加载样本 分类测试和训练集
def loadData(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as f:
        lines = csv.reader(f)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
                if random.random() < split:
                    trainingSet.append(dataset[x])   #生成随机数大于2/3用作training, 1/3 test, 2/3 training
                else:
                    testSet.append(dataset[x])

#计算距离
def computerDistance(instance1, instance2, length):
    '''
    :param instance1: 维度坐标
    :param instance2: 维度坐标
    :param length: 维度长度
    :return: distance
    '''
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]),2)
    return math.sqrt(distance)

#get一个测试集相邻最近的K个邻居
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):   #与所有的训练集进行距离计算
        dist = computerDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))    #将所有计算的距离进行排序 append最近的三个邻居
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#对K个邻居进行投票分类,少数服从多数
def getResponse(neighbors):
    votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in votes:   #进行投票
            votes[response] += 1
        else:
            votes[response] = 1
    # print(votes,'\n')
    sortVotes = sorted(votes.items(), key=operator.itemgetter(1),reverse=True)   #[('Iris-virginica', 3)]
    return sortVotes[0][0]


#计算准确率
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:   #测试集里的class是否与预测分类后的class一致
            correct += 1
    return (correct/float(len(testSet))) * 100

def main():
    trainingSet = []
    testSet = []
    split = 0.67
    k = 3
    loadData(r'./irisdata.txt', split=split, trainingSet=trainingSet, testSet=testSet)
    predictions = []
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('>predicted=' + repr(result) + ',actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet,predictions)
    print('Accuracy:' + repr(accuracy) + '%')

if __name__ == '__main__':
    main()