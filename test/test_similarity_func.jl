using LinearAlgebra

@testset "distance_euclidean" begin
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]

    expected = sqrt((4.0 - 1.0)^2 + (5.0 - 2.0)^2 + (6.0 - 3.0)^2)

    result = distance_euclidean(a, b)

    @test isapprox(result, expected; atol = 1.0e-12)
end

@testset "distance_manhattan" begin
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]

    expected = abs(4.0 - 1.0) + abs(5.0 - 2.0) + abs(6.0 - 3.0)

    result = distance_manhattan(a, b)

    @test isapprox(result, expected; atol = 1.0e-12)
end

@testset "local_lipschitz_constant" begin
    x = [
        1.0 5.0;
        2.0 6.0;
        3.0 7.0;
        4.0 8.0;
    ]
    a = [
        0.1 0.5;
        0.2 0.6;
        0.3 0.7;
        0.4 0.8;
    ]
    x_pert = [
        1.1 5.0;
        2.0 6.5;
        3.0 7.0;
        4.1 8.0;
    ]
    a_pert = [
        0.1 0.6;
        0.3 0.6;
        0.3 0.8;
        0.5 0.8;
    ]

    expected = [1.4142, 0.4]

    result = lipschitz_constant(
        a, a_pert, x, x_pert
    )

    @test isapprox(result, expected; atol = 1.0e-4)
end
