using ThreadPinning
pinthreads(:cores)
threadinfo()

using BenchmarkTools, NeuralOperators, Random, Optimisers, Zygote

rng = Xoshiro(1234)

train!(args...; kwargs...) = train!(MSELoss(), AutoZygote(), args...; kwargs...)

function train!(loss, backend, model, ps, st, data; epochs=10)
    l1 = loss(model, ps, st, first(data))

    tstate = Lux.Experimental.TrainState(model, ps, st, Adam(0.01f0))
    for _ in 1:epochs, (x, y) in data
        _, _, _, tstate = Lux.Experimental.single_train_step!(backend, loss, (x, y), tstate)
    end

    l2 = loss(model, ps, st, first(data))

    return l2, l1
end

# FNO
n_points = 128
batch_size = 64

x = rand(Float32, 1, n_points, batch_size);
y = rand(Float32, 1, n_points, batch_size);
data = [(x, y)];
t_fwd = zeros(5)
t_train = zeros(5)
for i in 1:5
    chs = (1, 128, fill(64, i)..., 128, 1)
    model = FourierNeuralOperator(gelu; chs=chs, modes=(16,))
    ps, st = Lux.setup(rng, model)
    model(x, ps, st) # TTFX

    t_fwd[i] = @belapsed $model($x, $ps, $st)

    t_train[i] = @belapsed train!($model, $ps, $st, $data; epochs=10)
end

println("\n## FNO")
print("| #layers | Forward | Train: 10 epochs | \n")
print("| --- | --- | --- | \n")
for i in 1:5
    print("| $i | $(t_fwd[i] * 1000) ms | $(t_train[i] * 1000) ms | \n")
end

# DeepONets
eval_points = 1
batch_size = 64
dim_y = 1
m = 32

u = rand(Float32, m, batch_size);
y = rand(Float32, dim_y, eval_points, batch_size);

g = rand(Float32, eval_points, batch_size);

data = [((u, y), g)]
t_fwd = zeros(5)
t_train = zeros(5)
for i in 1:5
    ch_branch = (m, fill(64, i)..., 128)
    ch_trunk = (dim_y, fill(64, i)..., 128)
    model = DeepONet(; branch=ch_branch, trunk=ch_trunk)
    ps, st = Lux.setup(rng, model)
    model((u, y), ps, st) # TTFX

    t_fwd[i] = @belapsed $model(($u, $y), $ps, $st)

    t_train[i] = @belapsed train!($model, $ps, $st, $data; epochs=10)
end

println("\n## DeepONet")
print("| #layers | Forward | Train: 10 epochs | \n")
print("| --- | --- | --- | \n")
for i in 1:5
    print("| $i | $(t_fwd[i] * 1000) ms | $(t_train[i] * 1000) ms | \n")
end
