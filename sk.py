from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy

# data_np=pd.DataFrame(iris_datasets.data,columns=['sepal length (cm)','sepal width (cm)', 'petal length (cm)','petal width (cm)'])
# label=pd.DataFrame(iris_datasets.target,columns=['target'])
# data_np['target']=label
# print(data_np.head(5))
# data_np[1:].to_csv("iris.csv")
#
# with open("iris.csv") as f:
#     for line in f:
#         split = line.split(",")
#         split[-1]=split[-1][0:1]
#         print(split)

#花萼长度,花萼宽度,花瓣长度,花瓣宽度


def data_split():
    iris_datasets = load_iris()
    iris_X = iris_datasets.data
    iris_Y = iris_datasets.target
    x_train, x_test, y_train, y_test = train_test_split(iris_X, iris_Y, test_size=0.4, random_state=0)
    #test不需要分类数据
    test=[]
    test_result=[]
    for i,j in zip(x_test,y_test):
        test.append(i)
        test_result.append(float(j))
        # test.append(numpy.append(i, j))
    df1 = pd.DataFrame()  # 写入csv
    result1 = df1.append(test, ignore_index=True)
    result1.to_csv('test.csv')
    test_re = df1.append(test_result, ignore_index=True)
    test_re.to_csv('testresult.csv')



    train = []
    for i, j in zip(x_train, y_train):
        train.append(numpy.append(i,j))
    df2 = pd.DataFrame()  # 写入csv
    result2 = df2.append(train, ignore_index=True)
    result2.to_csv('train.csv')

