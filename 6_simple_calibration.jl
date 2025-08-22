# =============================================================================
# Inverse Problem Example: Sinusoid
# =============================================================================
# 
# This notebook demonstrates how to solve a simple inverse problem using 
# EnsembleKalmanProcesses.jl, a Julia package for gradient-free parameter 
# estimation using ensemble-based methods.
# 
# Problem Setup
# 
# An inverse problem starts from a set of observed data `y`, an assumed forward 
# model `G(θ)`, and some model or measurement noise `Γ`. The goal is to infer 
# the unknown parameters `θ` that most likely produced the observations.
# 
# We assume the following observational model:
# 
# y = G(θ) + Γ
# 
# Our objective is to estimate the optimal parameters θ* such that:
# 
# G(θ*) ≈ y
# 
# In this example, our forward model is a sinusoidal function with an unknown 
# amplitude and vertical shift:
# 
# G(A, v) = A sin(φ + t) + v, ∀ t ∈ [0, 2π]
# 
# Here:
# - A is the amplitude
# - v is the vertical shift
# - φ is a fixed (but unknown) phase shift, treated as part of the noise
# - t is a vector of input values over [0, 2π]
# 
# Our observations `y` consist of summary statistics of the true signal: the 
# maximum - minimum and the mean. From these noisy statistics, we aim to infer 
# the underlying values of A and v.
# 
# We assume that the true parameters were drawn from:
# - A* ~ N(2, 1)
# - v* ~ N(0, 25)
# 
# Parameter Estimation with Ensemble Kalman Process (EKP)
# 
# The Ensemble Kalman Process (EKP) is an iterative, derivative-free algorithm 
# designed for solving inverse problems. It generalizes the Ensemble Kalman 
# Filter (EnKF) to parameter estimation, treating unknown parameters as latent 
# states to be inferred.
# 
# Key Advantages
# 
# - Derivative-free: No need to compute gradients of complex Earth system models
# - Handles nonlinearity: Works with highly nonlinear forward models
# - Quantifies uncertainty: Provides ensemble-based uncertainty estimates
# - Scalable: Efficient for high-dimensional parameter and data spaces
# 
# Algorithm Overview
# 
# 1. Initialize an ensemble of parameter vectors from prior distributions
# 2. Evaluate the forward model for each ensemble member
# 3. Compute empirical statistics (mean, covariance) of model outputs
# 4. Update parameters using Kalman-inspired rules based on observation mismatch
# 5. Iterate steps 2-4 until convergence
# 
# This method provides a practical approach to solving inverse problems in 
# scientific applications where the forward model is nonlinear or complex 
# (e.g. land modeling).

# =============================================================================
# Setup and Dependencies
# =============================================================================

ENV["JULIA_PKG_PRECOMPILE_AUTO"]=0
using Pkg
required_pkgs = ["EnsembleKalmanProcesses", "CairoMakie", "LinearAlgebra", "Random", "Distributions"]
Pkg.add(required_pkgs)
Pkg.instantiate()
Pkg.precompile()

using LinearAlgebra, Random
using Distributions, Plots
using CairoMakie
CairoMakie.activate!()
using EnsembleKalmanProcesses.ParameterDistributions
import EnsembleKalmanProcesses as EKP

rng_seed = 1234
rng = Random.MersenneTwister(rng_seed)

# =============================================================================
# Forward Model Definition
# =============================================================================
# 
# We define our sinusoidal model with a random phase shift to simulate model uncertainty:

dt = 0.01
time_range = 0:dt:(2 * pi + dt)
function model(amplitude, vert_shift)
    phi = 2 * pi * rand(rng)
    return amplitude * sin.(time_range .+ phi) .+ vert_shift
end

# =============================================================================
# Observation Map
# =============================================================================
# 
# The full forward map G maps from raw model output to observations consisting 
# of summary statistics:

function G(theta)
    theta, vert_shift = theta
    sincurve = model(theta, vert_shift)
    return [maximum(sincurve) - minimum(sincurve), mean(sincurve)]
end

# =============================================================================
# Now we generate our synthetic observations using our "true" parameters.
# =============================================================================

true_amplitude = 1.0
true_vertical_shift = 7.0
theta_true = (true_amplitude, true_vertical_shift)

Γ = 0.1 * I
noise_dist = MvNormal(zeros(2), Γ)
y = G(theta_true) .+ rand(noise_dist)

# =============================================================================
# Now we define our multivariate normal prior distribution using the 
# `constrained_gaussian` function. `constrained_gaussian` takes in a name, mean, 
# standard deviation, and bounds. The constraints refer to the bounds placed on 
# the prior distribution.
# 
# For the amplitude, we define a prior with mean 2 and standard deviation 1. 
# It is additionally constrained to be nonnegative. For the vertical shift we 
# define a Gaussian prior with mean 0 and standard deviation 5.
# =============================================================================

prior_u1 = constrained_gaussian("amplitude", 2, 1, 0, Inf)
prior_u2 = constrained_gaussian("vert_shift", 0, 5, -Inf, Inf)
prior = combine_distributions([prior_u1, prior_u2])

# =============================================================================
# We now generate the initial ensemble and set up the ensemble Kalman inversion. 
# We set a random seed to ensure reproducibility of our results.
# =============================================================================

N_ensemble = 20
N_iterations = 1000

initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)
ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, EKP.Inversion(); rng);

# =============================================================================
# Run the Inversion
# =============================================================================
# 
# We are now ready to carry out the inversion. At each iteration, we get the 
# ensemble from the last iteration, apply G(θ) to each ensemble member, and 
# apply the Kalman update to the ensemble.

for i in 1:N_iterations
    params_i = EKP.get_ϕ_final(prior, ensemble_kalman_process)

    G_ens = hcat([G(params_i[:, i]) for i in 1:N_ensemble]...)

    #controller = EKP.DataMisfitController(on_terminate = "continue")
    EKP.update_ensemble!(ensemble_kalman_process, G_ens)
end

# =============================================================================
# Results & Analysis
# =============================================================================
# 
# Now we can get the final ensemble, structured as a matrix where each row is 
# a different parameter and each column is a different ensemble member

final_ensemble = EKP.get_ϕ_final(prior, ensemble_kalman_process)

# =============================================================================
# We can also obtain just the mean parameter values:
# =============================================================================

final_mean = EKP.get_ϕ_mean_final(prior, ensemble_kalman_process)

# =============================================================================
# The final values are close to the true amplitude and vertical shift, and the 
# small spread in the full final ensemble indicates confidence in the final result.
# =============================================================================

# =============================================================================
# Visualization
# =============================================================================

# Create the figure and axis
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Time", ylabel = "Model output")

# Plot the true model
lines!(ax, time_range, model(theta_true...), color = :black, linewidth = 4, label = "Truth")

fig

# Plot the initial ensemble
for i in 1:N_ensemble
    lines!(
        ax,
        time_range,
        model(EKP.get_ϕ(prior, ensemble_kalman_process, 1)[:, i]...),
        color = :red,
        label = i == 1 ? "Initial ensemble" : ""
    )
end

# Plot the final ensemble
for i in 1:N_ensemble
    lines!(
        ax,
        time_range,
        model(final_ensemble[:, i]...),
        color = :blue,
        label = i == 1 ? "Final ensemble" : ""
    )
end

fig

save("fig.png", fig)
