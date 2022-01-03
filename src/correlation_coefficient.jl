using Random
using StatsBase
using Distributions


function sortby_x(x, y)
    sortedx = sortperm(x)
    (x[sortedx], y[sortedx])
end


function chatterjee_correlation_symmetric(x, y)
    max(chatterjee_correlation(x, y), chatterjee_correlation(y, x))
end


"""
    chatterjee_correlation(x, y)

Chatterjee correlation with limiting values [0, 1].
Describes how well y is a noiseless function of x -> non-symmetric by intention.
No restrictions on the law of (x, y).
"""
function chatterjee_correlation(x, y)
    randp = randperm(length(x))
    x, y = x[randp], y[randp]
    sortedx, sortedy_byx = sortby_x(x, y)
    length(unique(sortedx)) == length(sortedx) || @info "Ties in x. Breaking at random."

    ranks_y = ordinalrank(sortedy_byx)

    if length(unique(sortedy_byx)) == length(sortedy_byx) 
        return chatterjee_correlation_no_ties_y(sortedx, ranks_y)
    else
        @info "Ties in y."
        ranks_y_inv = ordinalrank(sortedy_byx; rev=true)
        return chatterjee_correlation_ties_y(sortedx, ranks_y, ranks_y_inv)
    end
end


function chatterjee_correlation_ties_y(sortedx, ranks_y, ranks_y_inv)
    nom = sum(abs.(diff(ranks_y)))
    denom = sum(ranks_y_inv .* (length(sortedx) .- ranks_y_inv))
    1.0 - (length(sortedx) .* nom) / (2.0 * denom)
end


function chatterjee_correlation_no_ties_y(sortedx, ranks_y)
    nom = sum(abs.(diff(ranks_y)))
    denom = length(sortedx)^2 - 1.0
    1.0 - (3.0 * nom) / denom
end


function max_value_chatterjee_correlation(x::AbstractVector, y::AbstractVector)
    @info "This is for no ties in Y. For ties in Y: not given."
    n = max(length(x), length(y))
    max_value_chatterjee_correlation(n)
end


max_value_chatterjee_correlation(n::Real) = (n-2)/(n+1)


minimum_value_chatterjee_correlation(x::AbstractVector, y::AbstractVector) = minimum_value_chatterjee_correlation(length(x))


"""
    minimum_value_chatterjee_correlation(n::Real)

Minimum value of Chatterjee correlation is -1/2 + O(1/n).
This value only occurs, when y's are sorted alternatingly.
Therefore, large negative correlation means: non i.i.d. sample.
"""
function minimum_value_chatterjee_correlation(n::Real)
    vals = sort(randn(n))
    alternating_vals = reduce(vcat, [[x, y] for (x, y) in zip(vals[1:n÷2], vals[(n÷2)+1:end])])
    chatterjee_correlation(1:length(alternating_vals), alternating_vals)
end


function p_value_asymptotic_null(chatterjee_cor, n)
    n < 20 || @info "n should not be smaller than 20 for the asymptotics to hold (see paper)."

    null_distribution = Normal(0, √(2.0/5.0))

    1 .- cdf(null_distribution, √(n).*chatterjee_cor)
end


function p_value_asymptotic_null_non_continuous(chatterjee_cor, n, ranks_y, ranks_y_inv)
    @assert n == length(ranks_y) "n (=$(n)) and length(ranks_y) (=$(length(ranks_y))) don't match."
    n < 20 || @info "n should not be smaller than 20 for the asymptotics to hold (see paper)."

    v = variance_asymptotic_null_non_contiuous(ranks_y, ranks_y_inv)
    null_distribution = Normal(0, √(v))

    1 .- cdf(null_distribution, √(n).*chatterjee_cor)
end


function chatterjee_std_dev_asymptotic(n)
    n < 20 || @info "n should not be smaller than 20 for the asymptotics to hold (see paper)."

    √(2.0/(5.0*n))
end


function chatterjee_std_dev_asymptotic_non_continuous(n, ranks_y, ranks_y_inv)
    @assert n == length(ranks_y) "n (=$(n)) and length(ranks_y) (=$(length(ranks_y))) don't match."
    n < 20 || @info "n should not be smaller than 20 for the asymptotics to hold (see paper)."

    v = variance_asymptotic_null_non_contiuous(ranks_y, ranks_y_inv)
    √(v/n)
end


function variance_asymptotic_null_non_contiuous(ranks_y, ranks_y_inv)
    n = length(ranks_y)
    
    sorted_ranks_y = sort(ranks_y)
    ind = 1:n
    ind2 = (2.0*n) .- 2.0 .* ind .+ 1
    ai = sum(ind2 .* sorted_ranks_y.^2)/(n^4)
    ci = sum(ind2 .* sorted_ranks_y)/(n^3)
    cq = cumsum(sorted_ranks_y)
    m = (cq .+ (n .- ind) .* sorted_ranks_y)
    b = sum(m.^2)/(n^5)
    CU = (1.0/n^3)*sum(ranks_y_inv .* (n .- ranks_y_inv))
    v = (ai - 2*b + ci^2)/(CU^2)
    return v
end
