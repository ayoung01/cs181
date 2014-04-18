import numpy as np
import matplotlib.pyplot as plt

epsilon = 0.00000001
v = np.zeros(117)
r = np.zeros([117,16])
policy = np.zeros(101)
c = {7:(1,1),12:(1,2),1:(1,3),14:(1,4),2:(2,1),13:(2,2),8:(2,3),11:(2,4),16:(3,1),3:(3,2),10:(3,3),5:(3,4),9:(4,1),6:(4,2),15:(4,3),4:(4,4)}
pg = [[0,0,0,0,0,0],[0,7,12,1,14,0],[0,2,13,8,11,0],[0,16,3,10,5,0],[0,9,6,15,4,0],[0,0,0,0,0,0]]
q = np.zeros([117,16])
f = True

def R(s,a):
	return 0.6*(int(s+a==101)-int(s+a>101)) + 0.1*((int(s+pg[c[a][0]-1][c[a][1]])==101)-int(s+pg[c[a][0]-1][c[a][1]]>101)) + 0.1*((int(s+pg[c[a][0]+1][c[a][1]])==101)-int(s+pg[c[a][0]+1][c[a][1]]>101)) + 0.1*((int(s+pg[c[a][0]][c[a][1]-1])==101)-int(s+pg[c[a][0]][c[a][1]-1]>101)) + 0.1*((int(s+pg[c[a][0]][c[a][1]+1])==101)-int(s+pg[c[a][0]][c[a][1]+1]>101))

k = 1

while f == True:
	v_old = (v.tolist())[:]
	f = False
	for s in range(0,101):
		for a in range(16):
			q[s][a] = R(s,a+1)+0.6*v_old[s+a+1]+0.1*v_old[s+pg[c[a+1][0]-1][c[a+1][1]]]+0.1*v_old[s+pg[c[a+1][0]+1][c[a+1][1]]]+0.1*v_old[s+pg[c[a+1][0]][c[a+1][1]-1]]+0.1*v_old[s+pg[c[a+1][0]][c[a+1][1]+1]]
			policy[s] = (q[s].tolist()).index(max(q[s]))
		v[s] = q[s][policy[s]]
		if abs(v[s]-v_old[s]) > epsilon:
			f = True
	print k
	k = k+1

print v
print policy

fig = plt.figure()
b1 = fig.add_subplot(121)
b2 = fig.add_subplot(122)
b1.bar(range(117),v.tolist(),1)
b2.bar(range(101),policy.tolist(),1)
plt.show()





