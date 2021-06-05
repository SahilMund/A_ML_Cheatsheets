import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=fig.gca(projection='3d')
x=np.arange(-5,5,0.25)
y=np.arange(-5,5,0.25)
x,y=np.meshgrid(x,y)
r=np.sqrt(x**2+y**2)
z=np.sin(r)
surf=ax.plot_surface(x,y,z,rstride=1,cstride=1,cmap=cm.BrBG)
plt.show()






















'''a,b,c=np.loadtxt('C:/Users/Sahil/Desktop/pans.csv',unpack=True,delimiter=',')
plt.plot(b,'gs')
#plt.scatter(b,c)
plt.title('csv data')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()'''


