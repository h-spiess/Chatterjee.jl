#%%
using RollingFunctions
using Random
using Statistics
using Test
using Distributions

#%%
linear(x=range(-1., 1, 1000); a=3., b=1., noisefn=(x) -> 0.1.*randn(length(x))) = (x, a .* x .+ b .+ noisefn(x))


function smooth_random_walk(x=randn(1000); cycles=8, windowspan=50)
    y = x
    for _ in 1:cycles
        y = runmean(y, windowspan)
    end
    (1:length(y), y)
end

"As described in the paper."
function two_gaussians(;n=200)
    gaussian_one = randn(2, n÷2)
    gaussian_two = randn(2, n÷2) .+ [5, 5]
    gaussians = [gaussian_one gaussian_two][:, randperm(n)]
    gaussians[1, :], gaussians[2, :]
end

@testset "invariant to strictly increasing transformations" begin
    for (x, y) in [smooth_random_walk(), linear()]
        @test chatterjee_correlation(x, y) == chatterjee_correlation(x, exp.(y))
        @test chatterjee_correlation(x, y) == chatterjee_correlation(x, 3.5*y .+ 2.)
    end
end


@testset "smooth random walk data" begin
    x, y = smooth_random_walk()

    # y is a function of x
    @test chatterjee_correlation(x, y) > chatterjee_correlation(y, x)
    
    # higher than linear correlation
    @test chatterjee_correlation(x, y) > cor(x, y)
end


@testset "linear data" begin
    x, y = linear(; noisefn=(x)->0.0)

    # y is a linear, noiseless function of x
    @test chatterjee_correlation(x, y) == chatterjee_correlation(y, x)

    x, y = linear(; noisefn=(x) -> 10.0 .* randn(length(x)))
    @test chatterjee_correlation(x, y) > chatterjee_correlation(y, x)
end

@testset "max min value chatterjee" begin
    n = 20
    @test max_value_chatterjee_correlation(n) ≈ 0.86 atol=0.1
    @test max_value_chatterjee_correlation(ones(n), ones(n)) ≈ 0.86 atol=0.1

    for n in [5, 10, 20, 100, 1000]
        @test max_value_chatterjee_correlation(n) <= 1.0
        @test max_value_chatterjee_correlation(ones(n), ones(n)) <= 1.0

        @test minimum_value_chatterjee_correlation(n) >= -0.5
        @test minimum_value_chatterjee_correlation(ones(n), ones(n)) >= -0.5
    end
end


@testset "null distribution p-value vs r-package XICOR" begin
    # asymptotic null for n > 20 and y continuous   
    # √n * chatterjee_correlation(x, y) -> N(0, 2/5) with mean, variance

    null_distribution = Normal(0, √(2.0/5.0))
    
    @test cdf(null_distribution, √(10).*0.55) ≈ 0.9970202 atol=0.001
    @test cdf(null_distribution, √(4).*0.75) ≈ 0.991147 atol=0.001
end

@testset "asymptotic null non continuous" begin
    @test variance_asymptotic_null_non_contiuous(1:1000, 1000:-1:1) ≈ 0.4 atol=0.01
end