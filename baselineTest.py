import baseline
import os


data_path = './BONC/'
data = os.listdir(data_path)
f = open('result.txt', 'w')
for i in data:
    apparatus = baseline.Apparatus(data_path + i)
    res = apparatus.angle
    try:
        apparatus.test()
        f.write(i + '\t长指针读数： %f 短指针读数： %f \n' % (res[0], res[1]))
    except IndexError as reason:
        f.write(i + str(reason) + '\n')
        continue



f.close()