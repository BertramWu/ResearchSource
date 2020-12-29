import torch
import time
import math
import matplotlib.pyplot as plt
import torch.utils.data as Data

in_put, hidden, out_put = 2, 100, 1
Nf, N_ic = 4000, 400
batch_size = 200
pi = math.pi


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
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, out_put)
        )

    def forward(self, x):
        x = self.model(x)
        return x

device = torch.device('cuda')  ### use GPU acceleration
X_f = torch.rand(Nf, 2)

X_f[:, 1] = X_f[:, 1] * 2 * pi
X_f[:500, 0] = X_f[:500, 0] / 4.0
X_f[500:1300, 0] = X_f[500:1300, 0] / 4.0 + 0.25
X_f[1300:2500, 0] = X_f[1300:2500, 0] / 4.0 + 0.5
X_f[2500:, 0] = X_f[2500:, 0] / 4.0 + 0.75

X_f_delay = torch.zeros(Nf, 2)
X_f_delay[:, 0] = X_f[:, 0] / 2.0
X_f_delay[:, 1] = X_f[:, 1]
X_f = X_f.to(device)  ###### X_f to GPU
X_f_delay = X_f_delay.to(device)  ################# X_f_delay to GPU
X_f = X_f.requires_grad_()
X_f_delay = X_f_delay.requires_grad_()
Loader = Data.DataLoader(dataset=Data.TensorDataset(X_f, X_f_delay), batch_size=batch_size, shuffle=True)

X_ic = torch.rand(N_ic, 2)
X_ic[:, 0] = 0.0
X_ic[:, 1] = X_ic[:, 1] * 2 * pi
X_ic = X_ic.to(device)  #########
X_ic = X_ic.requires_grad_()

net = NETWORK().to(device)  ##########
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
sheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300, 500, 700, 900],
                                                gamma=0.5, last_epoch=-1)
criterion = torch.nn.MSELoss()

start = time.time()  # time point
net.train()
for e in range(10 + 1):
    for batch_x, batch_x_delay in Loader:
        X0 = X_ic[:, 1].detach()
        X = batch_x[:, 1].detach()
        y_pred = net(batch_x)
        y_pred_delay = net(batch_x_delay)
        y_pred_ic = net(X_ic)
        y_pred_ic_t = torch.autograd.grad(y_pred_ic, [X_ic], grad_outputs=torch.ones_like(y_pred_ic),
                                          create_graph=True)[0][:, 0]
        order1 = torch.autograd.grad(y_pred, [batch_x], grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
        order2_tt = torch.autograd.grad(order1[:, 0], [batch_x], grad_outputs=torch.ones_like(order1[:, 0]),
                                        create_graph=True)[0][:, 0]
        order1_delay = torch.autograd.grad(y_pred_delay, [batch_x_delay], grad_outputs=torch.ones_like(y_pred_delay),
                                           create_graph=True)[0]
        order1_delay_t = order1_delay[:, 0]
        order2_delay_tt = torch.autograd.grad(order1_delay[:, 0], [batch_x_delay],
                                              grad_outputs=torch.ones_like(order1_delay_t), create_graph=True)[0][:, 0]
        loss = criterion(16.0 * order1_delay_t * order2_delay_tt + torch.cos(X) * order2_tt,
                         torch.zeros_like(order2_tt)) \
               + criterion(y_pred_ic, torch.zeros_like(y_pred_ic)) + \
               criterion(y_pred_ic_t, torch.cos(X0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if e % 10 == 0:
        print(e, loss.item())
print('Training time cost:', time.time() - start)

net.eval()
net = net.to('cpu')  #####
# torch.save(net.state_dict(),'E:/python/Jupyter/TrainedModels_Saving/nn_proportional_delay_PDE.pth')
torch.save(net.state_dict(),'./nn_proportional_delay_PDE.pth')
N_t, N_x = 100, 600
t = torch.linspace(0, 1, N_t)
x = torch.linspace(0, 2 * pi, N_x)
t, x = torch.meshgrid(t, x)
X_test = torch.zeros(N_t * N_x, 2)
for i in range(N_t):
    for j in range(N_x):
        X_test[i * N_x + j, :] = torch.tensor([t[i, j], x[i, j]])
Y_test = net(X_test).detach().numpy().reshape(N_t, N_x)
t = t.numpy()
x = x.numpy()

fig = plt.figure()
ax3 = plt.axes(projection='3d')
ax3.plot_surface(t, x, Y_test, rstride=1, cstride=1, cmap='rainbow')
ax3.set_xlabel('t')
ax3.set_ylabel('x')
ax3.set_zlabel('u(t,x)')
plt.show()