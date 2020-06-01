module RandomizedSearches
    import Statistics: mean, quantile
    import StatsBase
    using Base.Iterators
    using NearestNeighbors

    export init
    export run!
    export exhaust!

    mutable struct HyperParameters
        names::Array{Symbol}
        ranges::AbstractArray
        weights::AbstractArray
        n::Int
        ncombinations::Int
        combinations::AbstractArray
    end
    struct RandomizedSearchResult
        parameters::AbstractArray
        value::Number
    end
    mutable struct RandomizedSearch
        parameters::HyperParameters
        tried::Dict
        iter::Int
        iter_samples
        max_samples::Int
        max_iterations::Int
        patience::Int
        max_patience
        best
        top
        other::Dict
    end

    uniform_kernel(u::Real) = 1.0
    epanechnikov(u::Real) = 3//4 * (1 - u^2) * (abs(u) <= 1.0)
    tricube(u::Real) = 70//81 * (1 - abs(u)^3)^3 * (abs(u) <= 1.0)
    kernels=Dict{Symbol,Function}(
        :uniform_kernel=>uniform_kernel,
        :epanechnikov=>epanechnikov,
        :tricube=>tricube,
    )

    function init(names, ranges; iter_samples=[0.01], max_samples=Inf, max_iterations=10, max_patience=Inf)
        hparams_combinations=collect(product(ranges...))[1:end]
        weights=StatsBase.aweights(ones(length(hparams_combinations)))
        hparams=HyperParameters(names, ranges, weights, length(names), length(hparams_combinations), hparams_combinations)
        if maximum(iter_samples)<1.0
            iter_samples=[round(Int,i*hparams.ncombinations) for i in iter_samples]
        end
        max_samples = max_samples==Inf ? hparams.ncombinations : max_samples
        RandomizedSearch(hparams,Dict(), 1, iter_samples, max_samples, max_iterations, 0, max_patience, nothing, nothing, Dict())
    end

    function get_iter_samples(rs::RandomizedSearch)
        if rs.iter>length(rs.iter_samples)
            return rs.iter_samples[end]
        end
        rs.iter_samples[rs.iter]
    end

    function run_randomized_search_iteration!(rs::RandomizedSearch, f::Function; nbest=5, max_failed_tries=0.1)
        queue=[]
        failed_tries=0
        max_failed_tries=rs.parameters.ncombinations
        #Add highest probability
        if length(rs.tried)!=0
            candidate=rs.parameters.combinations[findmax(rs.parameters.weights)[2]]
            if ~(candidate in keys(rs.tried))
                push!(queue,candidate)
            end
        end
        #Add random
        while length(queue)<get_iter_samples(rs) && length(queue)+length(rs.tried) < rs.max_samples
            candidate=StatsBase.sample(rs.parameters.combinations, rs.parameters.weights)
            if candidate in keys(rs.tried) || candidate in queue
                failed_tries+=1
                if failed_tries==max_failed_tries
#                     display("Low probability draws only")
                    break
                end
                continue
            end
            push!(queue,candidate)
        end
        for candidate in queue
            rs.tried[candidate]=f(candidate...)
        end
        update_best!(rs; nbest=nbest)
        rs.iter+=1
    end

    function update_best!(rs::RandomizedSearch; nbest=5)
        sorted_tried=sort(collect(rs.tried), by=x->x[2])
        if length(sorted_tried)>0
            rs.best=RandomizedSearchResult(collect(sorted_tried[1][1]), sorted_tried[1][2])
            rs.top=[
                RandomizedSearchResult(collect(sorted_tried[i][1]), sorted_tried[i][2])
                for i=1:minimum([nbest,length(sorted_tried)])
            ]
            return rs.best
        end
        nothing
    end

    function run!(rs::RandomizedSearch, f::Function; kernel=:tricube, quant=0.05, kNN=3, kNNleafsize=10, alpha_weights=0.5, max_failed_tries=50, callbacks=[])
        #alpha_ewm smoothes weights on parameter dimension (0=>constant, 1=>no averaging)
        #alpha_weights smoothes weights between iterations (0=>do not update, 1=>update immediately)
        for i=1:rs.max_iterations
            prev_best=[]
            if length(rs.tried)!=0
                prev_best=rs.best.parameters
                try
                    # Sometimes this fails if tried is too short (e.g., after exhaust!)
                    update_weights!(rs; kernel=kernel, alpha_weights=alpha_weights, kNN=kNN, kNNleafsize=kNNleafsize,quant=quant)
                catch
                end
            end
            run_randomized_search_iteration!(rs, f; max_failed_tries=max_failed_tries)
            if rs.best.parameters==prev_best
                rs.patience+=1
            else
                rs.patience=0
            end
            if length(callbacks)!=0
                for ff in callbacks
                    ff(rs)
                end
            end
            if rs.patience==rs.max_patience
                break
            end
        end
    end

    function update_weights!(rs::RandomizedSearch; alpha_weights=0.5, kNN=3, kNNleafsize=10, kernel=:tricube, quant=0.5)
        kernel=kernels[kernel]
        tried=collect(rs.tried)
        tried_values=map(x->x[2], tried);
        minimum_tried_value = minimum(values(tried_values))

        tree = KDTree(1. .* hcat(map(x->collect(x[1]), tried)...); leafsize = kNNleafsize)
    #     tree = BallTree(1. .* hcat(map(x->collect(x[1]), tried)...); leafsize = kNNleafsize)

        boundary_tried_value = quantile(tried_values, quant)-minimum_tried_value

        interpolated_values=[
            begin
                o=0
                if (ps,) in keys(rs.tried)
                    o=rs.tried[ps]
                    print("+")
                else
                    ix_k_ps=knn(tree, [ps...], kNN)[1]
                    o=mean(tried_values[ix_k_ps])
                end
                o
            end
            for ps in rs.parameters.combinations
        ]

        new_weights=kernel.((interpolated_values.-minimum_tried_value)./boundary_tried_value)
        new_weights=(1-alpha_weights).*rs.parameters.weights.values .+ alpha_weights.*new_weights;
        rs.parameters.weights=StatsBase.aweights(new_weights)
    end

    function exhaust!(rs::RandomizedSearch, f::Function, ranges...)
        hparams_combinations=collect(product(ranges...))[1:end]
        for ps in hparams_combinations
            rs.tried[ps]=f(ps...)
        end
        update_best!(rs)
    end

end # module
