import timeit


# DeepONet

import_code = """import deepxde as dde
import torch
"""

n_iters = 100
fwd_timed_arr = [0.0] * 5
training_timed_arr = [0.0] * 5

for i in range(1, 6):
    setup_code = f"""eval_points = 64
batch_size = 64
dim_y = 1
m = 32

ch_branch = [m] + [64]*{i} + [128]
ch_trunk = [dim_y] + [64]*{i} + [128]
u = torch.rand(batch_size, m)
y = torch.rand(batch_size, dim_y)
output = torch.rand(batch_size, dim_y)

u_ = torch.rand(8, m)
y_ = torch.rand(8, dim_y)
output_ = torch.rand(8, dim_y)

data = dde.data.Triple(
    X_train= (u,y), y_train= output, X_test=(u_, y_), y_test= output_
)

net = dde.nn.DeepONet(
    ch_branch,
    ch_trunk,
    "tanh",
    "Glorot normal",
)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["mean l2 relative error"])
"""

    test_code = "model.predict((u,y))"
    timed = timeit.timeit(
        setup=import_code + setup_code, stmt=test_code, number=n_iters
    )
    fwd_timed_arr[i - 1] = timed

    test_code = "model.train(epochs = 10)"
    timed = timeit.timeit(
        setup=import_code + setup_code, stmt=test_code, number=n_iters
    )
    training_timed_arr[i - 1] = timed
    # print(i, "\t", timed/n_iters * 1000, " ms \n")

print("## DeepONet")
print("| #layers | Forward | Train: 10 epochs |")
print("| --- | --- | --- |")
for i in range(1, 6):
    print(
        "| ",
        i,
        " | ",
        fwd_timed_arr[i - 1] / n_iters * 1000,
        " ms | ",
        training_timed_arr[i - 1] / n_iters * 1000,
        " ms |",
    )


# FNO

import_code = """from neuralop.models import FNO1d
import torch
"""

timed_arr = [0.0] * 5
n_iters = 100
fwd_timed_arr = [0.0] * 5
training_timed_arr = [0.0] * 5

n_iters = 1000
for i in range(1, 6):
    setup_code = f"""operator1d = operator1d = FNO1d(n_modes_height=16, 
                    hidden_channels=64,
                    in_channels=1, 
                    out_channels=3,
                    n_layers= {i},
                    lifting_channels=128,
                    projection_channels=128)

batch_size = 64
n_points = 128
x = torch.rand(batch_size, 1, n_points)
y = torch.rand(batch_size, 3, n_points)
optimiser = torch.optim.Adam(operator1d.parameters(),lr=1e-4)
def train_model(model, data, y, optimser, epochs):
    loss = torch.mean((y- model(data))**2)
    loss.backward()
    optimiser.step()
    """
    test_code = "y = operator1d(x)"
    timed = timeit.timeit(
        setup=import_code + setup_code, stmt=test_code, number=n_iters
    )
    fwd_timed_arr[i - 1] = timed

    test_code = """train_model(operator1d, x, y, optimiser, 10)
    """
    timed = timeit.timeit(
        setup=import_code + setup_code, stmt=test_code, number=n_iters
    )
    training_timed_arr[i - 1] = timed


print("## FNO ")
print("| #layers | Forward | Train: 10 epochs | ")
print("| --- | --- | --- | ")
for i in range(1, 6):
    print(
        "| ",
        i,
        " | ",
        fwd_timed_arr[i - 1] / n_iters * 1000,
        " ms | ",
        training_timed_arr[i - 1] / n_iters * 1000,
        " ms | ",
    )
