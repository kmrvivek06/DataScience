import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_regions(X, y, classifier):
    markers = ('x','>','*')
    colors = ('red','blue','yellow','green')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    res = 0.02
    #Plot regions
    x1min, x1max = X[:,0].min() -1, X[:,0].max() + 1
    x2min, x2max = X[:,1].min() -1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1min,x1max,res),np.arange(x2min,x2max,res))

    output = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    output = output.reshape(xx1.shape)
    plt.figure(figsize=(8,8))
    plt.pcolormesh(xx1,xx2, output, cmap=plt.cm.coolwarm_r)
    #plt.contourf(xx1, xx2, output, alpha=0.4, cmap=cmap)
    #PLOT ALL SAMPLES
    for index, item in enumerate(np.unique(y)):
        plt.scatter(x=X[y == item, 0], y=X[y == item, 1],alpha=0.8, c=cmap(index),
        marker=markers[index], label=item)

        #plt.scatter(X[:,0],X[:,1], c=y, edgecolors='k',cmap=plt.cm.Paired,marker=markers,label=np.unique(y))
    plt.xlabel('petal length std')
    plt.ylabel('petal width std')

    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    plt.legend(loc='best',bbox_to_anchor=(0.5, 1.05),
          ncol=3,fancybox=True, shadow=True)
    plt.show()

