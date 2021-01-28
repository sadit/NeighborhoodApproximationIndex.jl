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
# Using the Neighborhood Approximation (NAPP) Index on CIFAR-10


This example shows how to create a NAPP over datasets of images easily.
"""

# ╔═╡ d8d27dbc-5507-11eb-20e9-0f16ddba080b
md"""
As first step, you must download the dataset (CIFAR10) before fetching the train data, e.g.

```CIFAR10.download()```

apparently, this doesn't work on Pluto and must be done using the terminal directly and accepting the terms and conditions of using the dataset.

You can also accept it from Pluto passing keyworkd argument `i_accept_the_terms_of_use=true`, e.g., `CIFAR10.download(i_accept_the_terms_of_use=true)`.


"""

# ╔═╡ cad95a24-5509-11eb-336f-91e5091503ec
# CIFAR10.download(i_accept_the_terms_of_use=true)

# ╔═╡ a23b0cae-455d-11eb-0e50-4dc31c050cc1
begin	
	T, y = CIFAR10.traindata()
	n = size(T, 4)
	X = [Float32.(view(T, :, :, :, i)) for i in 1:n]
	
	size(X)
	#length(X), size(X[1]), length(X[1])
end

# ╔═╡ 7d8db714-55c8-11eb-37a0-dd7e0744f6e1
function rgbimage(arr)
	A = Matrix{RGB}(undef, 32, 32)
	for i in 1:32, j in 1:32
		A[j, i] = RGB(arr[i, j, :]...)
	end

	A
end

# ╔═╡ 1ce583f6-54fb-11eb-10ad-b5dc9328ca3b
index = DeloneInvIndex(SqL2Distance(), X; numcenters=128, initial=:rand, maxiters=3, ksearch=3);
# index = fit(Sequential, X);

# ╔═╡ 5b743cbc-54fa-11eb-1be4-4b619e1070b2
begin
	sel = @bind example_symbol Slider(1:n, default=rand(1:n))
	distsel = @bind distname PlutoUI.Select([
			"L2" => "Euclidean",
			"L1" => "Manhattan",
			"LInf" => "Chebyshev",
			"p0.5" => "Minkowski with p=0.5",
			"Cos" => "Cosine"
			])
	
	ksel = @bind k html"<input type='range' min='1' max='15' step='1' default='3'>"
	
	sep = Matrix{RGB}(undef, 32, 6) # separator
	for i in eachindex(sep)
		sep[i] = RGB(1.0, 1.0, 1.0)
	end

	md"""
	
	query id: $(sel) |
	distance function: $(distsel)
	k: $(ksel)
	"""
end

# ╔═╡ ac0acb1c-618b-11eb-28f7-c966f3da7449
begin
	q = rgbimage(X[example_symbol]) # just to distinguish easily
	dist = Dict(
		"L2" => L2Distance(),
		"L1" => L1Distance(),
		"LInf" => LInftyDistance(),
		"p0.5" => LpDistance(0.5),
		"Cos" => CosineDistance(),
	)[distname]

	res = KnnResult(k)
	index_ = copy(index, dist=dist)
	with_terminal() do
		println("search time:")
		@time search(index_, X[example_symbol], res)
		
		for p in res
			print(p.id => p.dist, ", ")
		end
		println("end; k=$(length(res))")
	end
end

# ╔═╡ def63abc-45e7-11eb-231d-11d94709acd3
begin
	h = hcat(q, sep, [rgbimage(X[p.id]) for p in res]...)
	
	md"""

Query Id: $(example_symbol)
database size: $(length(X))
	
	
# Result:
$(h)
	
note: the blank space separates query from results

	"""
end

# ╔═╡ c1b7e0ac-54fd-11eb-2153-b599f834da36


# ╔═╡ Cell order:
# ╠═5cd87a9e-5506-11eb-2744-6f02144677ff
# ╠═c5af5d4a-455b-11eb-0b57-4d8d63615b85
# ╠═d8d27dbc-5507-11eb-20e9-0f16ddba080b
# ╠═cad95a24-5509-11eb-336f-91e5091503ec
# ╠═a23b0cae-455d-11eb-0e50-4dc31c050cc1
# ╠═7d8db714-55c8-11eb-37a0-dd7e0744f6e1
# ╠═1ce583f6-54fb-11eb-10ad-b5dc9328ca3b
# ╟─5b743cbc-54fa-11eb-1be4-4b619e1070b2
# ╟─ac0acb1c-618b-11eb-28f7-c966f3da7449
# ╟─def63abc-45e7-11eb-231d-11d94709acd3
# ╠═c1b7e0ac-54fd-11eb-2153-b599f834da36
