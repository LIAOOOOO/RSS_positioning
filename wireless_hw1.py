import pandas as pd
import math
import matplotlib.pyplot as plt

train = pd.read_csv('testfile.csv')
valid = pd.read_csv('valid.csv')
ans = [[0]*2709 for a in range(520)]
index = 0

for i in range(40):#test
    for j in range(2709):#train
        buff = 0
        for x in range(520):
            buff+=math.pow((valid.iloc[i,x]-train.iloc[j,x]), 2)

        ans[i][j] = math.sqrt(buff)
    print(i,j,ans[i])
#print(ans)

#1NN
lastans = []
adder_1NN = 0
for m in range(40):
    rank = [index for index, value in sorted(list(enumerate(ans[m])), key=lambda x: x[1])]
    n = rank[0]
    a = math.pow(train.iloc[n,520] - valid.iloc[m,520],2)
    b = math.pow(train.iloc[n,521] - valid.iloc[m,521],2)
    mini = math.sqrt(a+b)
    adder_1NN+=mini
    lastans.append(mini)  

print("average = ",adder_1NN/40)
plt.subplot(2,3,1)
plt.title('1NN') 
plt.axvline(adder_1NN/40, color='r', linestyle='--')
plt.hist(lastans,bins = 40,density=True)
plt.show



#3NN average
three_NN_average=[]
adder_3NNaver = 0
for m in range(40):
    rank = [index for index, value in sorted(list(enumerate(ans[m])), key=lambda x: x[1])]
    e = rank[0]
    f = rank[1]
    g = rank[2]
    t_x = (train.iloc[e,520]+train.iloc[f,520]+train.iloc[g,520])/3
    t_y = (train.iloc[e,521]+train.iloc[f,521]+train.iloc[g,521])/3
    a = math.pow(t_x - valid.iloc[m,520],2)
    b = math.pow(t_y - valid.iloc[m,521],2)
    medium_a = math.sqrt(a+b)
    adder_3NNaver+=medium_a
    three_NN_average.append(medium_a)
    
print("averagr_3NN = ",adder_3NNaver/40)
plt.subplot(2,3,2)
plt.title('3NN average') 
plt.axvline(adder_3NNaver/40, color='r', linestyle='--')
plt.hist(three_NN_average,bins = 40,density=True)
plt.show

#5NN average
fiveNN_average=[]  
adder_5NNaver = 0
for m in range(40):
    rank = [index for index, value in sorted(list(enumerate(ans[m])), key=lambda x: x[1])]
    e = rank[0]
    f = rank[1]
    g = rank[2]
    h = rank[3]
    i = rank[4]
    t_x = (train.iloc[e,520]+train.iloc[f,520]+train.iloc[g,520]+train.iloc[h,520]+train.iloc[i,520])/5
    t_y = (train.iloc[e,521]+train.iloc[f,521]+train.iloc[g,521]+train.iloc[h,521]+train.iloc[i,521])/5
    a = math.pow(t_x - valid.iloc[m,520],2)
    b = math.pow(t_y - valid.iloc[m,521],2)
    medium_a = math.sqrt(a+b)
    adder_5NNaver+=medium_a
    fiveNN_average.append(medium_a)
print("average_5NN = ",adder_5NNaver/40)

plt.subplot(2,3,3)
plt.title('5NN average') 
plt.axvline(adder_5NNaver/40, color='r', linestyle='--')
plt.hist(fiveNN_average,bins = 40,density=True)
plt.show


#3NN weight

three_NN_weight=[]
adder_3NNwei = 0
for m in range(40):
    rank = [index for index, value in sorted(list(enumerate(ans[m])), key=lambda x: x[1])]
    e = rank[0]
    f = rank[1]
    g = rank[2]
    
    weight=[1/ans[m][e], 1/ans[m][f], 1/ans[m][g]]
    weights = sum(weight)
    
    t_x = (train.iloc[e,520]/ans[m][e]+train.iloc[f,520]/ans[m][f]+train.iloc[g,520]/ans[m][g]) /weights
    t_y = (train.iloc[e,521]/ans[m][e]+train.iloc[f,521]/ans[m][f]+train.iloc[g,521]/ans[m][g]) /weights
    a = math.pow(t_x - valid.iloc[m,520],2)
    b = math.pow(t_y - valid.iloc[m,521],2)
    medium_a = math.sqrt(a+b)
    adder_3NNwei+=medium_a
    three_NN_weight.append(medium_a)
    
print("averagr_3NN_weight = ",adder_3NNwei/40)

plt.subplot(2,3,4)
plt.title('3NN weight') 
plt.axvline(adder_3NNwei/40, color='r', linestyle='--') 
plt.hist(three_NN_weight,bins = 40,density=True)
plt.show


#5NN weight
fiveNN_weight=[]
adder_5NNwei = 0
for m in range(40):
    rank = [index for index, value in sorted(list(enumerate(ans[m])), key=lambda x: x[1])]
    e = rank[0]
    f = rank[1]
    g = rank[2]
    h = rank[3]
    i = rank[4]
    weight=[1/ans[m][e], 1/ans[m][f], 1/ans[m][g], 1/ans[m][h], 1/ans[m][i] ]
    weights = sum(weight)
    
    t_x = (train.iloc[e,520]/ans[m][e] + train.iloc[f,520]/ans[m][f] + train.iloc[g,520]/ans[m][g] + train.iloc[h,520]/ans[m][h] + train.iloc[i,520]/ans[m][i]) /weights
    t_y = (train.iloc[e,521]/ans[m][e]+train.iloc[f,521]/ans[m][f]+train.iloc[g,521]/ans[m][g] + train.iloc[h,521]/ans[m][h] + train.iloc[i,521]/ans[m][i]) /weights
    a = math.pow(t_x - valid.iloc[m,520],2)
    b = math.pow(t_y - valid.iloc[m,521],2)
    medium_a = math.sqrt(a+b)
    adder_5NNwei+=medium_a
    fiveNN_weight.append(medium_a)   
print("averagr_5NN_weight = ",adder_5NNwei/40)



plt.subplot(2,3,5)
plt.title('5NN weight') 
plt.axvline(adder_5NNwei/40, color='r', linestyle='--')
plt.hist(fiveNN_weight,bins = 40,density=True)
plt.show