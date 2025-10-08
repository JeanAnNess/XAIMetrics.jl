using NPZ

@testset "Normalize Functions" begin
	a = npzread("npy_arrays/test_array_3x4.npy")
	expected = npzread("npy_arrays/result_normalize_by_max_abs.npy")
	normalized = normalize_by_max_abs(a)
	@test isapprox(normalized, expected; atol=1e-12)
end