# using NPZ

# @testset "Normalize Functions" begin
# 	a = npzread("npy_arrays/test_array_3x4.npy")
# 	expected = npzread("npy_arrays/result_normalize_by_max_abs.npy")
# 	normalized = normalize_by_max_abs(a)
# 	@test isapprox(normalized, expected; atol=1e-12)
# end

@testset "normalize_by_max_abs" begin
    @testset "Global Normalization (default)" begin
        a = [
            1.0 -2.0;
            3.0 -4.0
        ]

        expected = [
            0.25  -0.5;
            0.75 -1.0
        ]
        result = normalize_by_max_abs(a, nothing)
        @test result ≈ expected
    end
    @testset "Normalize along axis 1" begin
        # Note: Python axis 0 corresponds to Julia dim 1
        a = [
            1.0 -2.0;
            3.0 -4.0
        ]
        expected = [
            0.3333333333333333 -0.5;
            1.0                -1.0
        ]
        result = normalize_by_max_abs(a, [1])
        @test result ≈ expected
    end

    @testset "Normalize along axis 2" begin
        # Note: Python axis 1 corresponds to Julia dim 2
        a = [
            1.0 -2.0;
            3.0 -4.0
        ]
        expected = [
            0.5 -1.0;
            0.75 -1.0
        ]

        result = normalize_by_max_abs(a, [2])
        @test result ≈ expected
    end

    @testset "3D Normalize (Py axes 0, 1 -> Jl dims 1, 3)" begin
        # Python shape (N, H, C) = (2, 2, 2)
        # normalise_axes = [0, 1] (over N and H)

        # Julia equivalent shape (H, C, N) = (2, 2, 2)
        # normalise_axes = [1, 3]
        a = reshape([1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0], (2, 2, 2))

        expected = reshape([0.25, 0.5, 0.25, 0.5, 0.75, 1.0, 0.75, 1.0], (2, 2, 2))

        result = normalize_by_max_abs(a, [1, 3])
        @test result ≈ expected
    end

    @testset "All Zeros" begin
        a = [0.0 0.0; 0.0 0.0]
        expected = [0.0 0.0; 0.0 0.0]

        result = normalize_by_max_abs(a)
        @test result == expected
    end

end
