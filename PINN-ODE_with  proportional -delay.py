#solve y''(x) + y(x) - 5(y(x/2))^2   0,1,   -> y = e^(-2x)
import torch
import time
import  numpy as np
import  matplotlib.pyplot as plt

in_put,hidden,out_put = 1,32,1
N = 100
class NETWORK(torch.nn.Module):
    def __init__(self):
        super(NETWORK, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_put, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, out_put)
        )
    def forward(self, x):
        x = self.model(x)
        return x
X = torch.rand(N,in_put)*2.0-1.0
X_delay = torch.zeros_like(X)
X_delay = X/2.0
X = X.requires_grad_()
net = NETWORK()
optimizer = torch.optim.Adam(net.parameters(), lr=0.5e-3)
criterion = torch.nn.MSELoss()

epoches= 1000
loss_list = torch.zeros(epoches)
start = time.time()
for e in range(epoches):
    y_pred = net(X)
    y_pred_delay = net(X_delay)
    y0_pre = net(torch.tensor([-1.0]))
    y1_pre = net(torch.tensor([1.]))
    order1x = torch.autograd.grad(y_pred, [X], grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
    order2_xx = torch.autograd.grad(order1x, [X], grad_outputs=torch.ones_like(order1x), create_graph=True)[0]
    loss = criterion(order2_xx + y_pred - 5*y_pred_delay**2,torch.zeros_like(y_pred)) + \
           criterion(y0_pre,torch.exp(torch.tensor([2.0]))) + \
           criterion(y1_pre,torch.exp(torch.tensor([-2.0])))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if e % 10 == 0:
        print(e, loss)
    loss_list[e] = loss
print('The total training time:',time.time()-start)

x = torch.linspace(-1,1,100)
y = torch.squeeze(net(torch.unsqueeze(x,dim=1))).detach().numpy().reshape(100)
x = x.numpy()
fig,ax = plt.subplots(nrows=1,ncols=2)
fig.set_size_inches(10,4)
ax[0].grid(True,linestyle = '-.')
ax[0].plot(x,y,'r')
ax[0].plot(x,np.exp(-2*x),'bo',markersize = 1)
ax[0].set_xlabel('x',fontsize = 10)
ax[0].set_ylabel('y(x)',fontsize = 10)
ax[0].legend(['Prediction','Exact'],fontsize = 10)
ax[0].set_title('Compare Prediction with Exact value',fontsize = 12)
ax[0].tick_params(labelsize = 10,direction='in')
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)

e=np.linspace(1,epoches,epoches)
ax[1].grid(True,linestyle = '-.')
ax[1].plot(e,loss_list.detach().numpy(),'go',markersize = 2)
ax[1].set_title('The empirical loss function',fontsize=12)
ax[1].set_xlabel('Number of iteration steps',fontsize = 10)
ax[1].set_ylabel('Loss',fontsize = 10)
ax[1].tick_params(labelsize = 10,direction = 'in')
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
plt.savefig('F:/ChhromeDownload/acc-pred-epoches-loss.png',dpi = 300)
plt.show()