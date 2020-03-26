f1 = open('test.txt', 'r')
f2 = open('result.txt', 'r')
false = 0
for i in range(1001):
    r1 = f1.readline().split('数')
    r2 = f2.readline().split('数')
    try:
        if (r1[1][1:5] != r2[1][1:5] and (int(r1[1][1:5]) + 50) % 1200 != int(r2[1][1:5])) or (r1[2][1:3] != r2[2][1:3] and (int(r1[2][1:3]) + 1) % 100 != int(r2[2][1:3])):
            false += 1
            print('1_{0:0>4d}.jpg 出错！'.format(i))
    except:
        false += 1

print('错误率%f' % float(false/1001))
f1.close()
f2.close()
