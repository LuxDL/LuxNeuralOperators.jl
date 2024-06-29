module LuxNeuralOperators

using PrecompileTools: @recompile_invalidations

@recompile_invalidations begin
    using ArgCheck: @argcheck
    using ChainRulesCore: ChainRulesCore, NoTangent
    using ConcreteStructs: @concrete
    using FFTW: FFTW, irfft, rfft
    using Lux
    using LuxCore: LuxCore, AbstractExplicitLayer
    using NNlib: NNlib, ⊠
    using Random: Random, AbstractRNG
    using Reexport: @reexport
    using Integrals
end

const CRC = ChainRulesCore

@reexport using Lux

include("utils.jl")
include("transform.jl")

include("functional.jl")
include("layers.jl")

include("fno.jl")
include("deeponet.jl")
include("iko.jl")

export FourierTransform
export SpectralConv, OperatorConv, SpectralKernel, OperatorKernel
export FourierNeuralOperator
export DeepONet
export IntegralKernel, IntegralKernelOperator

end
