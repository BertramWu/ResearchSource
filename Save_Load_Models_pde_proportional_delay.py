import torch
from math import pi
import matplotlib.pyplot as plt
import numpy as np
in_put,hidden,out_put = 2,100,1
class NETWORK(torch.nn.Module):
    def __init__(self):
        super(NETWORK, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_put, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden,hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden,hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, out_put)
        )
    def forward(self, x):
        x = self.model(x)
        return x
model = NETWORK()

model.load_state_dict(torch.load("E:/python/Jupyter/TrainedModels_Saving/sinhtcosx_for_spare_use.pth"))

#plot 3-D figure
N_t,N_x = 500,100
t = torch.linspace(0,1.0,N_t)
x = torch.linspace(0,2*pi,N_x)
t,x = torch.meshgrid(t,x)
T_X= torch.zeros(N_t*N_x,2)
for i in range(N_t):
    for j in range(N_x):
        T_X[i*N_x+j,:] = torch.tensor([t[i,j],x[i,j]])
u = model(T_X).detach().numpy().reshape(N_t,N_x)

t = t.numpy()
x = x.numpy()
u_acc = np.sinh(t)*np.cos(x)

fig = plt.figure()
fig.set_size_inches(10,4)
ax0 = fig.add_subplot(1,2,1,projection = '3d')
ax0.plot_surface(t,x,u,rstride=1,cstride=1,cmap='rainbow')
# ax0.grid(b = False)
ax0.tick_params(labelsize = 10,direction='in')
ax0.set_xlabel('t',fontsize = 10)
ax0.set_ylabel('x',fontsize = 10)
ax0.set_title('Learned solution',fontsize = 10)
ax1 = fig.add_subplot(1,2,2,projection = '3d')
ax1.plot_surface(t,x,u_acc,rstride=1,cstride=1, cmap='rainbow')
# ax1.grid(b = None)
ax1.set_xlabel('t',fontsize = 10)
ax1.set_ylabel('x',fontsize = 10)
ax1.set_title('Exact solution',fontsize = 10)
plt.subplots_adjust(wspace=0)

plt.savefig('F:/ChhromeDownload/3dfigure-sinhxcosx.png',bbox_inches='tight',dpi = 300,meshgrid =False)

# plot heat map,100
# N_t ,N_x = 500,500
# t = torch.linspace(0,5,N_t)
# x = torch.linspace(0,2*pi,N_x)
# t,x = torch.meshgrid(t,x)
# T_X = torch.zeros(N_t*N_x,2)
# for i in range(N_t):
#     for j in range(N_x): T_X[i*N_x+j,:] = torch.tensor([t[i,j],x[i,j]])
# u = model(T_X).detach().numpy().reshape(N_t,N_x)
# fig2,ax2=plt.subplots(nrows=1,ncols=1)
# fig2.set_size_inches(5,4)
# im = ax2.imshow(u,cmap = 'rainbow',vmin = 0.0 ,vmax = 1.2)
# ax2.set_xticks(np.arange(0,501,100))
# ax2.set_yticklabels([0.0,'2pi/5','4pi/5','6pi/5','8pi/5','2pi'])
# ax2.set_yticks(np.arange(0,501,100))
# ax2.set_xticklabels([0,0.2,0.4,0.6,0.8,1.0])
# ax2.set_xlabel('t',fontsize = 10)
# ax2.set_ylabel('x',fontsize = 10)
# ax2.set_title('u(t,x)',fontsize = 10)
# ax2.tick_params(labelsize = 10,direction='in')
# zbar = fig2.colorbar(im,ax =ax2)
# zbar.minorticks_on()
# plt.savefig('F:/ChhromeDownload/heatmap-sinhtcosx.png',dpi = 500,meshgrid =False)

#plot comparison with accurance solution in 3 points
t1 = 0.25
t2 = 0.75
t3 = 0.5
x = np.linspace(0,2*pi,100)
T_X1 = torch.zeros(100,2)
T_X1[:,0] = 0.25
T_X1[:,1] = torch.linspace(0,2*pi,100)
T_X2 = torch.zeros(100,2)
T_X2[:,0] = 0.75
T_X2[:,1] = torch.linspace(0,2*pi,100)
T_X3 = torch.zeros(100,2)
T_X3[:,0] = 0.5
T_X3[:,1] = torch.linspace(0,2*pi,100)
u1 = model(T_X1).detach().numpy()
u2 = model(T_X2).detach().numpy()
u3 = model(T_X3).detach().numpy()
fig3,ax3 = plt.subplots(ncols=3)
fig3.set_size_inches(10,4)
ax3[0].plot(x,u1,'ro',markersize=2)
ax3[0].plot(x,np.sinh(t1)*np.cos(x),'b',markersize = 2)
ax3[0].set_xlabel('t')
ax3[0].set_title('t=0.25')
ax3[0].set_ylabel('u(t,x)')
ax3[0].legend(['Learned','Exact'],fontsize = 10)
ax3[0].grid(True,linestyle = '-.')
ax3[1].plot(x,u3,'r')
ax3[1].plot(x,np.sinh(t3)*np.cos(x),'bo',markersize= 2)
ax3[1].set_xlabel('t')
ax3[1].set_title('t=0.5')
ax3[1].grid(True,linestyle = '-.')
ax3[1].legend(['Learned','Exact'],fontsize = 10)
ax3[2].plot(x,u2,'r')
ax3[2].plot(x,np.sinh(t2)*np.cos(x),'bo',markersize = 2)
ax3[2].legend(['Learned','Exact'],fontsize = 10)
ax3[2].set_xlabel('t')
ax3[2].set_title('t=0.75')
ax3[2].grid(True,linestyle = '-.')
ax3[2].set_ylabel('u(t,x)')
plt.savefig('F:/ChhromeDownload/acc_pred3-sinhtcosx.png',dpi = 300,meshgrid =False)
plt.show()