using ThreadPinning
pinthreads(:cores)
threadinfo()

using BenchmarkTools, NeuralOperators, Random, Zygote
using Flux
using Optimisers: Adam

function train!(model::L, data, opt_state; epochs = 10, loss = Flux.Losses.mse) where { L<: FourierNeuralOperator}
    for epoch in 1:epochs
        Flux.train!(model, data, opt_state) do m, x, y
          loss(m(x), y)
        end
      end
end


function train!(model::L, data, opt_state; epochs = 10, loss = Flux.Losses.mse) where {L <: DeepONet}
    for epoch in 1:epochs
        Flux.train!(model, data, opt_state) do m, u, y, g
          loss(m(u, y), g)
        end
      end
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
    model = FourierNeuralOperator(; ch=chs, modes=(16,), σ=gelu)
    model(x) # TTFX

    t_fwd[i] = @belapsed $model($x)

    opt_state = Flux.setup(Adam(), model)

    t_train[i] = @belapsed train!($model, $data, $opt_state)
end

println("\n## FNO (Flux NeuralOperators.jl)")
print("| #layers | Forward | Train: 10 epochs | \n")
print("| --- | --- | --- | \n")
for i in 1:5
    print("| $i | $(t_fwd[i] * 1000) ms | $(t_train[i] * 1000) ms | \n")
end

# DeepONets
eval_points = 64
batch_size = 64
dim_y = 1
m = 32

u = rand(Float32, m, batch_size);
y = rand(Float32, dim_y, eval_points);

g = rand(Float32, batch_size, eval_points);

data = [(u, y, g)]
t_fwd = zeros(5)
t_train = zeros(5)
for i in 1:5
    ch_branch = (m, fill(64, i)..., 128)
    ch_trunk = (dim_y, fill(64, i)..., 128)
    model = DeepONet(ch_branch, ch_trunk)
    model(u, y) # TTFX

    t_fwd[i] = @belapsed $model($u, $y)

    opt_state = Flux.setup(Adam(), model)

    t_train[i] = @belapsed train!($model, $data, $opt_state)
end

println("\n## DeepONet (Flux NeuralOperators.jl)")
print("| #layers | Forward | Train: 10 epochs | \n")
print("| --- | --- | --- | \n")
for i in 1:5
    print("| $i | $(t_fwd[i] * 1000) ms | $(t_train[i] * 1000) ms | \n")
end
