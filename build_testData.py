f = open('test.txt', 'w')
a, b = -50, -1 # 长短指针初始读数
for i in range(1001):
    if int(i % 19) == 0:
        a = a + 50 if a // 1150 != 1 else 0 
    if int(i % 9.12) == 0:
        b = b + 1 if b // 99 != 1 else 0
    name = '1_{0:0>4d}.jpg     长指针读数：{1}     短指针读数：{2}\n'.format(i, a, b)
    f.write(name)
f.close()
