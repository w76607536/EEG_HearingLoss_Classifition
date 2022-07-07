'''
Description: 
Author: Wu Yubo
Date: 2022-05-19 00:21:22
LastEditTime: 2022-05-19 23:18:06
LastEditors:  
'''
# coding:utf-8
import matplotlib.pyplot as plt

data_dir = './models/9chan_2patients_5classes_RestNet18_woDS/log.csv'
Train_Loss_list = []
Train_Accuracy_list = []
Valid_Loss_list = []
Valid_Accuracy_list = []
f1 = open(data_dir, 'r')
data = []
# 把训练结果输出到result.txt里，比较笨的办法，按字节位数去取数字结果
for line in f1:
    #print(line.find("lr")," ",line)
    if line[0] == 'e':

        continue

    else:
        line_data = line.split(",")
        Train_Loss_list.append(float(line_data[2]))
        Train_Accuracy_list.append(float(line_data[3]))
        Valid_Loss_list.append(float(line_data[4]))
        Valid_Accuracy_list.append(float(line_data[5]))
f1.close()

fig = plt.figure(figsize = (14,5))
# 迭代了30次，所以x的取值范围为(0，30)，然后再将每次相对应的准确率以及损失率附在x上
x1 = range(len(Train_Loss_list))
x2 = range(len(Train_Loss_list))
y1 = Train_Accuracy_list
y2 = Train_Loss_list
y3 = Valid_Accuracy_list
y4 = Valid_Loss_list
plt.subplot(1,2,2)
plt.plot(x1, y1, 'o-',color='r')
plt.plot(x1, y1, 'o-', label="Train_Accuracy")
plt.plot(x1, y3, 'o-', label="Valid_Accuracy")
plt.title('Train accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.legend(loc='best')
#plt.savefig("train_results_loss.png")
plt.subplot(1,2,1)
plt.plot(x2, y2, '.-', label="Train_Loss")
plt.plot(x2, y4, '.-', label="Valid_Loss")
plt.title('Train loss vs. epoches')
plt.ylabel('Test loss')
plt.legend(loc='best')
picName = data_dir.split('/')[-2]
plt.savefig(picName+".png")
plt.show()



