### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ c5af5d4a-455b-11eb-0b57-4d8d63615b85
begin
	using MLDatasets
	using SimilaritySearch
	using NeighborhoodApproximationIndex
	using Colors
	using PlutoUI
end

# ╔═╡ 5cd87a9e-5506-11eb-2744-6f02144677ff
md"""
# Using the Neighborhood Approximation (NAPP) Index with MNIST


This example shows how to create a NAPP over datasets of images easily.
"""

# ╔═╡ d8d27dbc-5507-11eb-20e9-0f16ddba080b
md"""
As first step, you must download the dataset (MNIST, FashionMNIST) before fetching the train data, e.g.

```MNIST.download()```

apparently, this doesn't work on Pluto and must be done using the terminal directly and accepting the terms and conditions of using the dataset.

You can also accept it from Pluto passing keyworkd argument `i_accept_the_terms_of_use=true`, e.g., `MNIST.download(i_accept_the_terms_of_use=true)`.


"""

# ╔═╡ a23b0cae-455d-11eb-0e50-4dc31c050cc1
begin
	T, y = MNIST.traindata()
	n = size(T, 3)
	X = [Float32.(view(T, :, :, i)) for i in 1:n]
	length(X), size(X[1]), length(X[1])
end

# ╔═╡ 1ce583f6-54fb-11eb-10ad-b5dc9328ca3b
index = DeloneInvIndex(L2Distance(), X; numcenters=128, initial=:rand, maxiters=3, ksearch=3);
# index = fit(Sequential, X);

# ╔═╡ 9899d2b8-550c-11eb-3eef-59e40dfe6d26


# ╔═╡ 5b743cbc-54fa-11eb-1be4-4b619e1070b2
begin
	sel = @bind example_symbol html"<input type='range' min='1' max='$n' step='1'>"
	md"""
	select the query object using the bar: $(sel)
	"""
end

# ╔═╡ 3fb931fc-618a-11eb-2d13-eb5322bb307e
begin
	qinverted = 1 .- X[example_symbol]' # just to distinguish easily
	res = KnnResult(10)
	with_terminal() do
		@info "search time:"
		@time search(index, X[example_symbol], res)
		for p in res
			print(p.id => round(p.dist, digits=3), ", ")
		end
		println("end; k=$(length(res))")
	end
end

# ╔═╡ def63abc-45e7-11eb-231d-11d94709acd3
begin
		h = hcat(qinverted,  [X[p.id]' for p in res]...);

	md""" $(size(h))

Query Id: $(example_symbol)
database size: $(length(X))
	
	
# Result:
$(Gray.(h))


note: the symbol is the query object and its colors has been inverted
"""
end

# ╔═╡ Cell order:
# ╠═5cd87a9e-5506-11eb-2744-6f02144677ff
# ╠═c5af5d4a-455b-11eb-0b57-4d8d63615b85
# ╠═d8d27dbc-5507-11eb-20e9-0f16ddba080b
# ╠═a23b0cae-455d-11eb-0e50-4dc31c050cc1
# ╠═1ce583f6-54fb-11eb-10ad-b5dc9328ca3b
# ╠═9899d2b8-550c-11eb-3eef-59e40dfe6d26
# ╟─5b743cbc-54fa-11eb-1be4-4b619e1070b2
# ╟─3fb931fc-618a-11eb-2d13-eb5322bb307e
# ╠═def63abc-45e7-11eb-231d-11d94709acd3
