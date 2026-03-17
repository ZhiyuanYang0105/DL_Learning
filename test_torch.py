import sys, torch
print('exe:', sys.executable)
print('torch:', torch.__version__)
print('mps:', torch.backends.mps.is_available())
print('cuda:', torch.cuda.is_available())

x = torch.randn(4, 8)
y = torch.randn(4, 1)
model = torch.nn.Sequential(torch.nn.Linear(8,16), torch.nn.ReLU(), torch.nn.Linear(16,1))
opt = torch.optim.SGD(model.parameters(), lr=1e-2)
loss_fn = torch.nn.MSELoss()

pred = model(x)
loss = loss_fn(pred, y)
loss.backward()
opt.step()
print('one step ok, loss=', loss.item())