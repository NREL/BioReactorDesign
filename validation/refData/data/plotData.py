import numpy as np
import matplotlib.pyplot as plt
import time

xco2 = np.loadtxt('xco2_all.txt', delimiter=',')
gh = np.loadtxt('GasHoldup_all.txt', delimiter=',')


gh_exp17 = gh[:18]
gh_exp19 = gh[18:34]
gh_exp17_ngu = gh[34:54]
gh_exp17_hassaniga = gh[54:70]
gh_exp19_ngu = gh[70:86]
gh_exp19_hassaniga = gh[86:]


xco2_exp17 = xco2[:12]
xco2_exp19 = xco2[12:24]
xco2_exp17_ngu = xco2[24:40]
xco2_exp17_hassaniga = xco2[40:57]
xco2_exp19_ngu = xco2[57:70]
xco2_exp19_hassaniga = xco2[70:]


np.savez('val2_data.npz', 
         gh_exp17=gh_exp17,
         gh_exp19=gh_exp19,
         gh_exp17_ngu=gh_exp17_ngu,
         gh_exp17_hassaniga=gh_exp17_hassaniga,
         gh_exp19_ngu=gh_exp19_ngu,
         gh_exp19_hassaniga=gh_exp19_hassaniga,
         xco2_exp17=xco2_exp17,
         xco2_exp19=xco2_exp19,
         xco2_exp17_ngu=xco2_exp17_ngu,
         xco2_exp17_hassaniga=xco2_exp17_hassaniga,
         xco2_exp19_ngu=xco2_exp19_ngu,
         xco2_exp19_hassaniga=xco2_exp19_hassaniga)



#fig = plt.figure()
#plt.plot(xco2_exp17[:,0], xco2_exp17[:,1], 's', color='k')
#plt.plot(xco2_exp19[:,0], xco2_exp19[:,1], '^', color='k')
#plt.plot(xco2_exp17_ngu[:,0], xco2_exp17_ngu[:,1], color='k')
#plt.plot(xco2_exp17_hassaniga[:,0], xco2_exp17_hassaniga[:,1], '--', color='k')
#plt.plot(xco2_exp19_ngu[:,0], xco2_exp19_ngu[:,1], color='k')
#plt.plot(xco2_exp19_hassaniga[:,0], xco2_exp19_hassaniga[:,1], '--', color='k')
#plt.show()

#fig = plt.figure()
#plt.plot(gh_exp17[:,0], gh_exp17[:,1], 's', color='k')
#plt.plot(gh_exp19[:,0], gh_exp19[:,1], '^', color='k')
#plt.plot(gh_exp17_ngu[:,0], gh_exp17_ngu[:,1], color='k')
#plt.plot(gh_exp17_hassaniga[:,0], gh_exp17_hassaniga[:,1], '--', color='k')
#plt.plot(gh_exp19_ngu[:,0], gh_exp19_ngu[:,1], color='k')
#plt.plot(gh_exp19_hassaniga[:,0], gh_exp19_hassaniga[:,1], '--', color='k')
#plt.show()
