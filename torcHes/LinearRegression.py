import torch as th
import torch.nn as nn

#let y= 2x1 + 3x2
X_train = th.tensor([[1,1],[1,2],[2,1],[2,2],[2,3]], dtype=th.float32)
Y_train = th.tensor([[5],[8],[7],[10],[13]], dtype=th.float32)

n_samples, n_features = X_train.shape

X_test = th.tensor([6,7], dtype=th.float32)

input_size = n_features
output_size = Y_train[0].shape[0]

model = nn.Linear(input_size, output_size)

print(f'Prediction before training: f(6,7) = {model(X_test).item()}')

learning_rate = 0.075
n_iters = 1225

loss = nn.MSELoss()
optimizer = th.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters) :
    
    l = loss(model(X_train), Y_train)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    # if epoch % 10 == 0:
    #     print('epoch ', epoch+1, ': w = ', [*model.parameters()][0][0], ' loss = ', l)

print(f'Prediction after training: f(6,7) = {round(model(X_test).item(),2)}')