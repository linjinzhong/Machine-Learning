import pylab as pl
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.metrics import auc, roc_curve
from math import log, exp, sqrt

evaluate_result = "./ROC_data.txt"
db = [] 
pos, neg = 0, 0

with open(evaluate_result, 'r') as fs:
    for line in fs:
        # label:真实样本标签，score：二分类中预测为正样本的标签
        label, score = line.strip().split()
        if label[0] == '0':
            nonclk = 1  
            clk = 0
        else:
            clk = 1
            nonclk = 0
        score = float(score)
        db.append([score, nonclk, clk])
        pos += clk
        neg += clk



# 按预测得分从高到低排序
db = sorted(db, key = lambda x: x[0], reverse=True)

# 计算ROC坐标点
#预测：     正  负    
#实际： 正  tp  fn
#实际： 负  fp  tn
# x = tp / (tp + fn)  正样本召回率
# y = fp / (fp + tn)  1-负样本召回率
xy_arr = []
tp, fp = 0, 0
# 从高到低排序好的score中
# 以当前i位置所在score为阈值预测，之前遍历的预测为正样本，之后未遍历的为负样本
# 所以在当前阈值下的tp和fp可如下计算
for i in range(len(db)):
    tp += db[i][2]  #将正样本预测为正样本的数量
    fp += db[i][1]  #将负样本预测为正样本的数量
    xy_arr.append([fp / neg, tp / pos])



# 计算曲线下的面积AUC
AUC, prev_x = 0., 0
for x, y in xy_arr:
    if x != prev_x:
        AUC += (x - prev_x) * y
        prev_x = x
print("the auc is %s." % AUC)


# plot
x = [_v[0] for _v in xy_arr]
y = [_v[1] for _v in xy_arr]
xx = np.array(x)
yy = np.array(y)
Auc = auc(xx,yy)
pl.title("ROC cure of %s (AUC = %.4f)" % ("Taxi dateset", AUC))
pl.xlabel("False Positive Rate")
pl.ylabel("True Positive Rate")
pl.plot(x, y)
pl.show()


# 使用sklearn自带函数
x = [_v[0] for _v in xy_arr]
y = [_v[1] for _v in xy_arr]
xx = np.array(x)
yy = np.array(y)
Auc = auc(xx,yy)
db = np.array(db)
fpr, tpr, thresholds = roc_curve(db[:,2],db[:,0])
plt.figure()
plt.plot(fpr, tpr, color = 'darkorange', lw = 2, marker = 'o', label = 'ROC curve (AUC = %0.4f)' % Auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('ROC of Taxi dataset')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.show()