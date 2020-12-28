#u_t = a^2u_xx + b^2*u_xx(t-tau,x)
import torch
import time
import  matplotlib.pyplot as plt
import torch.utils.data as Data
in_put,hidden,out_put = 2,200,1
Nf,N_ic,N_bc =4000,800,1000
batch_size = 128
tau = 1.0
a ,b =0.2,0.02
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

device = torch.device('cuda')

X_f = torch.rand(Nf,2)
X_f[:,0] = X_f[:,0]*4.0 + 1.0
X_f_delay = torch.zeros(Nf,2)
X_f_delay[:,0] = X_f[:,0] - tau
X_f_delay[:,1] = X_f[:,1]
X_f = X_f.to(device)
X_f_delay = X_f_delay.to(device)
X_f = X_f.requires_grad_()
X_f_delay = X_f_delay.requires_grad_()
# Loader_x = Data.DataLoader(dataset=X_f,batch_size = batch_size,shuffle=False)
# Loader_x_delay = Data.DataLoader(dataset=X_f_delay,batch_size = batch_size,shuffle=False)
Loader = Data.DataLoader(dataset=Data.TensorDataset(X_f,X_f_delay),batch_size=batch_size,shuffle=False)

X_ic = torch.rand(N_ic,2)
X_ic = X_ic.to(device)
X_ic = X_ic.requires_grad_()

X_bc = torch.rand(N_bc,2)
X_bc[:,0] = X_bc[:,0]*5.0
X_bc[0:500,1] = 0.0
X_bc[500:,1] = 1.0
X_bc=X_bc.to(device)
X_bc = X_bc.requires_grad_()

net = NETWORK().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.5e-3)
sheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[200,400,600,800,900],gamma = 0.7,last_epoch=-1)
criterion = torch.nn.MSELoss()

def phi(X_ic):
    return torch.sin(X_ic[:,0])*X_ic[:,1]*(1.0-X_ic[:,1])

start = time.time()
net.train()
for e in range(1000 + 1):
    for batch_x,batch_x_delay in Loader:
        y_pred = net(batch_x)
        y_pred_delay = net(batch_x_delay)
        y_pred_ic = net(X_ic)
        y_pred_bc = net(X_bc)
        order1 = torch.autograd.grad(y_pred,[batch_x],grad_outputs=torch.ones_like(y_pred),create_graph=True)[0]
        order1t = order1[:,0]
        order2_xx = torch.autograd.grad(order1[:,1], [batch_x], grad_outputs=torch.ones_like(order1[:,1]),
                                        create_graph=True)[0][:,1]
        order1_delay = torch.autograd.grad(y_pred_delay,[batch_x_delay],grad_outputs=torch.ones_like(y_pred_delay),
                                             create_graph=True)[0]
        order2_delay_x = torch.autograd.grad(order1_delay[:,1],[batch_x_delay],grad_outputs=torch.ones_like(
                                             order1_delay[:,1]),create_graph=True)[0][:,1]
        loss = criterion(order1t - a**2 * order2_xx - b**2 * order2_delay_x,torch.zeros_like(order1t)) \
               + criterion(y_pred_ic,torch.unsqueeze(phi(X_ic),dim=1)) + \
                criterion(y_pred_bc,torch.zeros_like(y_pred_bc))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if e % 10 == 0:
        print(e, loss.item())
print('The running time:',time.time()-start)

net.eval()
net = net.to('cpu')  #####
torch.save(net.state_dict(),'E:/python/Jupyter/TrainedModels_Saving/series_training_model1.pth')
N_t,N_x = 500,100
t = torch.linspace(0,5,N_t)
x = torch.linspace(0,1,N_x)
t,x = torch.meshgrid(t,x)
X_test = torch.zeros(N_t*N_x,2)
for i in range(N_t):
    for j in range(N_x):
        X_test[i*N_x+j,:] = torch.tensor([t[i,j],x[i,j]])
Y_test = net(X_test).detach().numpy().reshape(N_t,N_x)
t = t.numpy()
x = x.numpy()

fig = plt.figure()
ax3 = plt.axes(projection = '3d')
ax3.plot_surface(t,x,Y_test,rstride=1,cstride=1,cmap='rainbow')
ax3.set_xlabel('t')
ax3.set_ylabel('x')
ax3.set_zlabel('u(t,x)')
plt.show()