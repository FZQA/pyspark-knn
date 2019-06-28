def judge(realfile,testfile):
    real_result = {}
    with open(realfile) as f:
        next(f)
        for line in f:
            split = line.split(",")
            real_result[int(split[0])] = float(split[1][:-1])
    test_result = {}
    with open(testfile) as f:
        for line in f:
            split = line.split(",")
            test_result[int(split[0][1:])] = float(split[1][1:-2])
            # test_result[split[0]]=split[1]
        # print(test_result)
    incorrect_num = 0
    for key in real_result:
        if real_result[key] != test_result[key]:
            incorrect_num+=1
    return incorrect_num


if __name__ == "__main__":
    incor_num= judge("testresult.csv","./knn-result5/part-00000")


