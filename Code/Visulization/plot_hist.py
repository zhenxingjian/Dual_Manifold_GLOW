import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

a = np.load('../data/results.npz')
odf_ = a['odf']
rodf_ = a['rodf']
odf_ = odf_.squeeze()
rodf_ = rodf_.squeeze()
error2 = (odf_**2-rodf_**2)**2
error2 = error2.sum(-1).sum(-1).sum(-1).sum(-1)
error2 = error2/10046.22

plt.hist(error2, 20, color='gray',ec='black')
mean = error2.mean()
std = error2.std()

print(mean)
print(std)

plt.plot([mean-std, mean, mean+std], [23,23,23], 'k')
plt.plot([mean], [23], 'ok')
plt.plot([mean-std,mean-std], [23-0.1, 23+0.1],'k')
plt.plot([mean+std,mean+std], [23-0.1, 23+0.1],'k')
ax = plt.gca()
ax.set_xlabel('Reconstruction error', fontsize = 20)
ax.set_ylabel('# of subjects', fontsize = 20)
plt.ticklabel_format(axis='x', style='sci', scilimits=(-4,-4))

plt.savefig('hist.png',bbox_inches='tight', dpi=300)

plt.show()