"""
    OperatorConv(ch::Pair{<:Integer, <:Integer}, modes::NTuple{N, <:Integer},
        ::Type{TR}; init_weight=glorot_uniform,
        permuted::Val{perm}=Val(false)) where {N, TR <: AbstractTransform, perm}

## Arguments

  - `ch`: A `Pair` of input and output channel size `ch_in => ch_out`, e.g. `64 => 64`.
  - `modes`: The modes to be preserved. A tuple of length `d`, where `d` is the dimension of
    data.
  - `::Type{TR}`: The transform to operate the transformation.

## Keyword Arguments

  - `init_weight`: Initial function to initialize parameters.
  - `permuted`: Whether the dim is permuted. If `permuted = Val(false)`, the layer accepts
    data in the order of `(ch, x_1, ... , x_d, batch)`. Otherwise the order is
    `(x_1, ... , x_d, ch, batch)`.

## Example

```jldoctest
julia> OperatorConv(2 => 5, (16,), FourierTransform{ComplexF32});

julia> OperatorConv(2 => 5, (16,), FourierTransform{ComplexF32}; permuted=Val(true));

```
"""
@concrete struct OperatorConv{perm, T <: AbstractTransform} <: AbstractExplicitLayer
    in_chs::Int
    out_chs::Int
    prod_modes::Int
    tform::T

    init_weight

    name::String
end

function LuxCore.initialparameters(rng::AbstractRNG, layer::OperatorConv)
    in_chs, out_chs = layer.in_chs, layer.out_chs
    scale = real(one(eltype(layer.tform))) / (in_chs * out_chs)
    return (;
        weights=scale * layer.init_weight(
            rng, eltype(layer.tform), out_chs, in_chs, layer.prod_modes))
end

@inline function LuxCore.parameterlength(layer::OperatorConv)
    return layer.prod_modes * layer.in_chs * layer.out_chs
end

function OperatorConv(ch::Pair{<:Integer, <:Integer}, modes::NTuple{N, <:Integer},
        ::Type{TR}; init_weight=glorot_uniform,
        permuted::Val{perm}=Val(false)) where {N, TR <: AbstractTransform{<:Number}, perm}
    name = "OperatorConv{$(string(nameof(TR)))}($(ch[1]) => $(ch[2]), $modes; permuted = $perm)"
    return OperatorConv{perm}(ch..., prod(modes), TR(modes), init_weight, name)
end

function (conv::OperatorConv{true})(x::AbstractArray, ps, st)
    return operator_conv(x, conv.tform, ps.weights), st
end

function (conv::OperatorConv{false})(x::AbstractArray, ps, st)
    N = ndims(conv.tform)
    xᵀ = permutedims(x, (ntuple(i -> i + 1, N)..., 1, N + 2))
    yᵀ = operator_conv(xᵀ, conv.tform, ps.weights)
    y = permutedims(yᵀ, (N + 1, 1:N..., N + 2))
    return y, st
end

"""
    SpectralConv(args...; kwargs...)

Construct a `OperatorConv` with `FourierTransform{ComplexF32}` as the transform. See
[`OperatorConv`](@ref) for the individual arguments.

## Example

```jldoctest
julia> SpectralConv(2 => 5, (16,));

julia> SpectralConv(2 => 5, (16,); permuted=Val(true));

```
"""
SpectralConv(args...; kwargs...) = OperatorConv(
    args..., FourierTransform{ComplexF32}; kwargs...)

"""
    OperatorKernel(ch::Pair{<:Integer, <:Integer}, modes::Dims{N}, transform::Type{TR},
        act::A=identity; allow_fast_activation::Bool=false, permuted::Val{perm}=Val(false),
        kwargs...) where {N, TR <: AbstractTransform, perm, A}

## Arguments

  - `ch`: A `Pair` of input and output channel size `ch_in => ch_out`, e.g. `64 => 64`.
  - `modes`: The modes to be preserved. A tuple of length `d`, where `d` is the dimension of
    data.
  - `::Type{TR}`: The transform to operate the transformation.

## Keyword Arguments

  - `σ`: Activation function.
  - `permuted`: Whether the dim is permuted. If `permuted = Val(true)`, the layer accepts
    data in the order of `(ch, x_1, ... , x_d , batch)`. Otherwise the order is
    `(x_1, ... , x_d, ch, batch)`.

All the keyword arguments are passed to the [`OperatorConv`](@ref) constructor.

## Example

```jldoctest
julia> OperatorKernel(2 => 5, (16,), FourierTransform{ComplexF64});

julia> OperatorKernel(2 => 5, (16,), FourierTransform{ComplexF64}; permuted=Val(true));

```
"""
function OperatorKernel(ch::Pair{<:Integer, <:Integer}, modes::Dims{N}, transform::Type{TR},
        act::A=identity; allow_fast_activation::Bool=false, permuted::Val{perm}=Val(false),
        kwargs...) where {N, TR <: AbstractTransform{<:Number}, perm, A}
    act = allow_fast_activation ? NNlib.fast_act(act) : act
    l₁ = perm ? Conv(map(_ -> 1, modes), ch) : Dense(ch)
    l₂ = OperatorConv(ch, modes, transform; permuted, kwargs...)

    return @compact(; l₁, l₂, activation=act, dispatch=:OperatorKernel) do x::AbstractArray
        l₁x = l₁(x)
        l₂x = l₂(x)
        @return @. activation(l₁x + l₂x)
    end
end

"""
    SpectralKernel(args...; kwargs...)

Construct a `OperatorKernel` with `FourierTransform{ComplexF32}` as the transform. See
[`OperatorKernel`](@ref) for the individual arguments.

## Example

```jldoctest
julia> SpectralKernel(2 => 5, (16,));

julia> SpectralKernel(2 => 5, (16,); permuted=Val(true));

```
"""
function SpectralKernel(ch::Pair{<:Integer, <:Integer}, modes::Dims{N},
        act::A=identity; kwargs...) where {N, A}
    return OperatorKernel(ch, modes, FourierTransform{ComplexF32}, act; kwargs...)
end
