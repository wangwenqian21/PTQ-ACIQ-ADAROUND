import math
def diedai_mub(x, mb=0, M=8):
    return math.exp(-x+mb)*3*(2**(2*M+2))

alpha_laplace_positive = {0: 1.86, 1: 2.83, 2: 3.89, 3: 5.02, 4: 6.2, 5: 7.41, 6: 8.64, 7: 9.89, 8: 11.16269}
alpha_accurate = {4:6.20476, 8:11.16269}
print('===== 当 mean/b=0 时，可计算出来的最优值alpha使 diedai_mub(alpha)-alpha 产生的差值 =====')
for k,v in alpha_laplace_positive.items():
    print(diedai_mub(v, M = k)-v)


### 测试M=4或8时我们给出的拟合函数情况 ###
print('===== mean/b在多项式拟合下的差值 =====')
M = 8

import matplotlib.pyplot as plt
import numpy as np

xlist = np.arange(-3,3,0.05).tolist()
ll = []

print(diedai_mub(alpha_accurate[M], M = M)-alpha_accurate[M])

def tiao(d): # 4bit
    k = -0.92
    return k*(0.996*d+0.011*abs(d)**0.65+0.001*d**2-0.0045)
def tiao2(d): # 8bit
    k = -0.89
    return k*(0.963*d+0.019*abs(d)**1.3+0.001*d**2-0.008)

for d in xlist:
    if M == 8:
        ll.append(diedai_mub(alpha_accurate[M]-tiao2(d), d, M = M)-(alpha_accurate[M]-tiao2(d)))
    elif M == 4:
        ll.append(diedai_mub(alpha_accurate[M]-tiao(d), d, M = M)-(alpha_accurate[M]-tiao(d)))
print(min(ll),max(ll))
plt.plot(xlist, ll)
plt.savefig(f'alpha_correct{M}')

print('===== 使用存储的b/mean数据计算|mean/b|<=3的比例 =====')
with open('bumdeeplabv3.json','r')as fp:
    import json
    allmub = json.load(fp)
county = 0
countall = len(allmub)
for mub in allmub:
    if abs(1/mub)<=3:
        county+=1

print(county/countall)