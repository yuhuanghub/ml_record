from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing,tree
from sklearn.externals.six import StringIO
import csv

with open('./MatterTree.csv','r') as TreeData:
    reader = csv.reader(TreeData)
    headers = next(reader)
    labelList = []
    featureList = []
    for row in reader:
        labelList.append(row[len(row) - 1])
        resultDict = {}
        for i in range(1,len(row) - 1):
            resultDict[headers[i]] = row[i]
        featureList.append(resultDict)

vec = DictVectorizer()
featureX = vec.fit_transform(featureList).toarray()

label = preprocessing.LabelBinarizer()
labelY = label.fit_transform(labelList)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(featureX,labelY)

with open('./grapTree.dot','w') as f:
    f = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)

