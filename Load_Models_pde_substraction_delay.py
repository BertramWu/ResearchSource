import torch
import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
in_put,hidden,out_put = 2,200,1
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

# model.load_state_dict(torch.load("E:/python/Jupyter/TrainedModels_Saving/series_training_model1.pth"))
model.load_state_dict(torch.load("./nn_subtraction_delay_PDE.pth"))

#plot 3-D figure
N_t,N_x = 500,100
t = torch.linspace(0,5,N_t)
x = torch.linspace(0,1,N_x)
t,x = torch.meshgrid(t,x)
X_test = torch.zeros(N_t*N_x,2)
for i in range(N_t):
    for j in range(N_x):
        X_test[i*N_x+j,:] = torch.tensor([t[i,j],x[i,j]])
Y_test = model(X_test).detach().numpy().reshape(N_t,N_x)

t = t.numpy()
x = x.numpy()

fig = plt.figure()
fig.set_size_inches(5,4)
ax = plt.axes(projection = '3d')
ax.plot_surface(t,x,Y_test,rstride=1,cstride=1,cmap='rainbow')
ax.grid(b = None,linestyle='-.')
ax.tick_params(labelsize = 10,direction='in')
ax.set_xlabel('t',fontsize = 10)
ax.set_ylabel('x',fontsize = 10)
ax.set_title('u(t,x)',fontsize = 10)
# plt.savefig('F:/ChhromeDownload/3dfigure.png',dpi = 300,meshgrid =False)

# plot heat map,100
N_t ,N_x = 500,500
t = torch.linspace(0,5,N_t)
x = torch.linspace(0,1,N_x)
t,x = torch.meshgrid(t,x)
X_test = torch.zeros(N_t*N_x,2)
for i in range(N_t):
    for j in range(N_x): X_test[i*N_x+j,:] = torch.tensor([t[i,j],x[i,j]])
Y_test = model(X_test).detach().numpy().reshape(N_t,N_x)

fig2,ax2=plt.subplots(nrows=1,ncols=1)
fig2.set_size_inches(5,4)
im = ax2.imshow(Y_test,cmap = 'rainbow',vmin = 0.0 ,vmax = 0.2)
ax2.set_xticks(np.arange(0,501,100))
ax2.set_yticklabels([0,1.0,2.0,3.0,4.0,5.0])
ax2.set_yticks(np.arange(0,501,100))
ax2.set_xticklabels([0,0.2,0.4,0.6,0.8,1.0])
ax2.set_xlabel('x',fontsize = 10)
ax2.set_ylabel('t',fontsize = 10)
ax2.set_title('u(t,x)',fontsize =10)
ax2.tick_params(labelsize = 10,direction='in')
zbar = fig2.colorbar(im,ax =ax2)
zbar.minorticks_on()
# plt.savefig('F:/ChhromeDownload/heatmap.png',dpi = 500,meshgrid =False)

#plot accurate figure T:0-5 X:0-1
# df = pd.read_csv('F:/private/SCIFiles/PINN/series solution.csv')
df = pd.read_csv('./series solution.csv')
array = df[2:].values
numarray = np.zeros_like(array)
for i in range(array.shape[0]):
    for j in range(array.shape[1]):
        numarray[i,j] = float(array[i,j])
prepared = np.zeros((51,4))
prepared[11:] = numarray
prepared[:11,0] = np.linspace(0,1,11)
prepared[:11,1] = np.sin(prepared[:11,0])*0.25*(1-0.25)
prepared[:11,2] = np.sin(prepared[:11,0])*0.5*(1-0.5)
prepared[:11,3] = np.sin(prepared[:11,0])*0.75*(1-0.75)

N_t= 51
x1,x2,x3 = 0.25,0.5,0.75
t = torch.linspace(0,5,N_t)
T_X = torch.zeros(N_t,2)
T_X[:,0] = t
T_X[:,1] = x1
u1 = model(T_X).detach().numpy()
T_X[:,1] = x2
u2 = model(T_X).detach().numpy()
T_X[:,1] = x3
u3 = model(T_X).detach().numpy()
fig1,ax1 = plt.subplots(ncols=3)
fig1.set_size_inches(10,4)
ax1[0].plot(t.numpy(),u1,'bo',markersize=3)
ax1[0].plot(t.numpy(),prepared[:,1],'r')
ax1[0].set_title('x=0.25')
ax1[0].set_xlabel('t')
ax1[0].set_ylabel('u(t,x)')
ax1[0].legend(['Series solution ','Learned value'],fontsize = 8)
ax1[0].grid(True,linestyle = '-.')

ax1[1].plot(t.numpy(),prepared[:,2],'bo',markersize = 3)
ax1[1].plot(t.numpy(),u2,'r')
ax1[1].set_xlabel('t')
ax1[1].set_title('x=0.5')
ax1[1].grid(True,linestyle = '-.')
ax1[1].legend(['Series solution ','Learned value'],fontsize = 8)
ax1[2].plot(t.numpy(),prepared[:,3],'bo',markersize = 3)
ax1[2].plot(t.numpy(),u3,'r')
ax1[2].set_title('x=0.75')
ax1[2].set_xlabel('t')
ax1[2].grid(True,linestyle = '-.')
ax1[2].legend(['Series solution ','Learned value'],fontsize = 8)
# plt.savefig('F:/ChhromeDownload/acc_pred3-pde-subtraction-delay.png',dpi = 300,meshgrid =False)
plt.show()