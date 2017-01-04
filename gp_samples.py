import numpy as np
import matplotlib.pyplot as plt

def rq_covariance(params, x, xp):
    h= params[0]
    alpha = params[1]
    l = params[2]
    diffs = np.expand_dims(x /l, 1) - np.expand_dims(xp/l, 0)
    return h**2 *np.power(1+np.sum(diffs**2, axis=2)/(2*alpha*l**2),(-1*alpha)) 

def plot_gp(ax,params,plot_xs,n_samples = 10,xlab=False,ylab=False):
    plot_xs = np.reshape(np.linspace(-5, 5, 300), (300,1))
    sampled_funcs = np.random.multivariate_normal(np.ones(len(plot_xs)), rq_covariance(params,plot_xs,plot_xs),\
    					size=10)
    ax.plot(plot_xs, sampled_funcs.T)
    ax.set_title(r'$\alpha = {},\/ l = {} $'.format(\
        params[1],params[2]),fontsize = 22)
    if xlab:
        ax.set_xlabel(r'$x$',fontsize = 20)
    if ylab:
        ax.set_ylabel(r'$f(x)$',fontsize = 20)

fig = plt.figure(figsize=(20,8), facecolor='white')
ax_1 = fig.add_subplot(231, frameon=False)
ax_2 = fig.add_subplot(232, frameon=False)
ax_3 = fig.add_subplot(233, frameon=False)
ax_4 = fig.add_subplot(234, frameon=False)
ax_5 = fig.add_subplot(235, frameon=False)
ax_6 = fig.add_subplot(236, frameon=False)
ax_1.set_xticks([])
ax_1.set_yticks([])
ax_2.set_xticks([])
ax_2.set_yticks([])
ax_3.set_xticks([])
ax_3.set_yticks([])
ax_4.set_xticks([])
ax_4.set_yticks([])
ax_5.set_xticks([])
ax_5.set_yticks([])
ax_6.set_xticks([])
ax_6.set_yticks([])

plot_xs = np.reshape(np.linspace(-5, 5, 300), (300,1))
plot_gp(ax_1,np.array([1,1,.5]),plot_xs,ylab=True)
plot_gp(ax_2,np.array([1,1,1.0]),plot_xs)
plot_gp(ax_3,np.array([1,1,2.0]),plot_xs)
plot_gp(ax_4,np.array([1,.02,1]),plot_xs,ylab=True,xlab=True)
plot_gp(ax_5,np.array([1,.1,1]),plot_xs,xlab=True)
plot_gp(ax_6,np.array([1,2.0,1]),plot_xs,xlab=True)
plt.savefig('gp_samples.png', format='png',bbox_inches='tight')