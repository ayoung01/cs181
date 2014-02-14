import numpy as np
import Image

a = np.load("cluster_centers.npy")

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

c = np.zeros((16,32,32,3))

for j in range (16):
	for i in range(1024):
	    c[j][i/32][i%32][0] = a[j][i] * 255
	    c[j][i/32][i%32][1] = a[j][i+1024] * 255
	    c[j][i/32][i%32][2] = a[j][i+2048] * 255

i = Image.fromarray(np.uint8(c[0]),'RGB')
i.save('c0.png')
i = Image.fromarray(np.uint8(c[1]),'RGB')
i.save('c1.png')
i = Image.fromarray(np.uint8(c[2]),'RGB')
i.save('c2.png')
i = Image.fromarray(np.uint8(c[3]),'RGB')
i.save('c3.png')
i = Image.fromarray(np.uint8(c[4]),'RGB')
i.save('c4.png')
i = Image.fromarray(np.uint8(c[5]),'RGB')
i.save('c5.png')
i = Image.fromarray(np.uint8(c[6]),'RGB')
i.save('c6.png')
i = Image.fromarray(np.uint8(c[7]),'RGB')
i.save('c7.png')
i = Image.fromarray(np.uint8(c[8]),'RGB')
i.save('c8.png')
i = Image.fromarray(np.uint8(c[9]),'RGB')
i.save('c9.png')
i = Image.fromarray(np.uint8(c[10]),'RGB')
i.save('c10.png')
i = Image.fromarray(np.uint8(c[11]),'RGB')
i.save('c11.png')
i = Image.fromarray(np.uint8(c[12]),'RGB')
i.save('c12.png')
i = Image.fromarray(np.uint8(c[13]),'RGB')
i.save('c13.png')
i = Image.fromarray(np.uint8(c[14]),'RGB')
i.save('c14.png')
i = Image.fromarray(np.uint8(c[15]),'RGB')
i.save('c15.png')



