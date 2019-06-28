# from pyspark import SparkContext
# from sklearn import datasets
# iris = datasets.load_iris()
# logFile = "a.txt"
# sc = SparkContext("local", "first app")
# logData = sc.textFile(logFile).cache()
# numAs = logData.filter(lambda s: 'a' in s).count()
# numBs = logData.filter(lambda s: 'b' in s).count()
# print("Line with a:%i,lines with b :%i" % (numAs, numBs))

# from pyspark import SparkConf, SparkContext # 创建SparkConf和SparkContext
# conf = SparkConf().setMaster("local").setAppName("lichao-wordcount")
# sc = SparkContext(conf=conf)
# # 输入的数据
# data=["hello","world","hello","word","count","count","hello"] # 将Collection的data转化为spark中的rdd并进行操作
# rdd=sc.parallelize(data)
# resultRdd = rdd.map(lambda word: (word,1)).reduceByKey(lambda a,b:a+b) # rdd转为collecton并打印
# resultColl = resultRdd.collect()
# for line in resultColl:
#     print(line) # 结束 sc.stop()

import pyspark
import heapq
import operator as op
from sk import data_split
from correct import judge
import shutil


def distanceSquared (a, b):
    dis = sum((x1-x2)**2 for (x1, x2) in zip(a, b))
    # print(dis)
    return dis


# 训练数据处理    训练数据比测试数据多了类别
def simple_csv_loader_train (line):
    split = line.split(",")
    return (int(split[0]), ((float(split[1]), float(split[2]),float(split[3]),float(split[4])),float(split[5])))


# 测试数据处理
def simple_csv_loader_test (line):
    split = line.split(",")
    # print(split)
    return (int(split[0]), (float(split[1]), float(split[2]),float(split[3]),float(split[4])))


# p: (id<p>, sample<p>)
# samples: [(id<s>, (sample<s>,class<s>,))]
# return: (id<p>, class<p>)
def kNNWorker (p, samples):
    # 找最小n个元素
    nsm = heapq.nsmallest(K, samples, key=lambda s: distanceSquared(s[1][0], p[1]))
    # print("nsm",nsm)
    c = {}
    for (_, (_, cls)) in nsm:
        c[cls] = c.get(cls, 0) + 1
    # print(c)
    best_class = max(c.items(), key=op.itemgetter(1))[0]
    return (p[0], best_class)


# training_rdd: RDD([(id<s>, (sample<s>,class<s>))])
# testing_rdd: RDD([(id<p>, sample<p>)])
# return: RDD([(id<p>, class<p>)])
def kNN (sc, training_rdd, testing_rdd):
    broad_training = sc.broadcast(training_rdd.collect())
    result = testing_rdd.map(lambda p: kNNWorker(p, broad_training.value))
    return result


# rdd: RDD([(id, (class, sample))])
# return: RDD([(id, (class<true>, class<pred>))])
def testKNN (sc, rdd, split=0.5, seed=None):
    train, test = rdd.randomSplit([split, 1-split], seed)
    hiddenTest = test.mapValues(op.itemgetter(1))
    knnOut = kNN(sc, train, hiddenTest)
    return test.mapValues(op.itemgetter(0)).join(knnOut)


def main(sc, training_set, testing_set, out_path):
    training_data = sc.textFile(training_set)
    head1 = training_data.take(1)[0]
    training_data = training_data.filter(lambda line: line != head1).map(simple_csv_loader_train)
    testing_data = sc.textFile(testing_set)
    head2 = testing_data.take(1)[0]
    testing_data = testing_data.coalesce(1,True).filter(lambda line: line != head2).map(simple_csv_loader_test)

    result = kNN(sc, training_data, testing_data)
    # top_k_res = result.takeOrdered(20)
    # print(top_k_res)
    result.saveAsTextFile(out_path)   # 保存一个文件 只能保存到文件夹
    return None


if __name__ == "__main__":
    data_split()
    sc = pyspark.SparkContext()
    for K in range(4,13):
        try:
            shutil.rmtree("./knn-result"+str(K))
        except:
            pass
        sc.broadcast(K)
        main(sc,"train.csv","test.csv","knn-result"+str(K))
        incorrect_num=judge("testresult.csv", "./knn-result"+str(K)+"/part-00000")
        print("\nK=",K,"预测错误的个数", incorrect_num)


