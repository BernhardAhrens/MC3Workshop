#!/usr/bin/env julia

# Global full land (snow+soil+canopy) run
# 
# The code sets up the ClimaLand land model on a spherical domain,
# forcing with ERA5 data, but does not actually run the simulation.
# To run the simulation, we strongly recommend using a GPU.

println("Setting up ClimaLand global land simulation...")

# First we import a lot of packages
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = 0
required_pkgs = ["CairoMakie", "ClimaAnalysis", "GeoMakie", "Printf", "StatsBase", "ClimaComms", "ClimaParams", "Dates", "ClimaUtilities", "Interpolations", "ClimaDiagnostics", "ClimaLand"]

import Pkg
Pkg.Registry.update()
Pkg.add(required_pkgs)

import ClimaComms
ClimaComms.@import_required_backends
import ClimaParams as CP
using Dates
using ClimaUtilities
import ClimaUtilities.TimeVaryingInputs:
    TimeVaryingInput, LinearInterpolation, PeriodicCalendar
import Interpolations
using ClimaDiagnostics
import ClimaLand
import ClimaLand.Parameters as LP
import ClimaLand.Simulations: LandSimulation, solve!
using Dates

using CairoMakie, ClimaAnalysis, GeoMakie, Printf, StatsBase
import ClimaLand.LandSimVis as LandSimVis

println("ClimaLand version: ", pkgversion(ClimaLand))

# Set the simulation float type, determine the
# context (MPI or on a single node), and device type (CPU or GPU).
# Create a default output directory for diagnostics.
const FT = Float64
context = ClimaComms.context()
ClimaComms.init(context)
device = ClimaComms.device()
device_suffix = device isa ClimaComms.CPUSingleThreaded ? "cpu" : "gpu"
root_path = "land_longrun_$(device_suffix)"
diagnostics_outdir = joinpath(root_path, "global_diagnostics")
outdir = ClimaUtilities.OutputPathGenerator.generate_output_path(diagnostics_outdir)
earth_param_set = LP.LandParameters(FT)

println("Using device: $device_suffix")
println("Output directory: $outdir")

# Set timestep, start_date, stop_date
Δt = 450.0
start_date = DateTime(2008)
stop_date = DateTime(2009)

println("Simulation period: $start_date to $stop_date")
println("Timestep: $Δt seconds")

# Create the domain
nelements = (101, 15)
domain = ClimaLand.Domains.global_domain(FT; context, nelements)

println("Domain created with $nelements elements")

# Low-resolution forcing data from ERA5 is used here,
# but high-resolution should be used for production runs.
println("Loading ERA5 forcing data...")
era5_ncdata_path = ClimaLand.Artifacts.era5_land_forcing_data2008_path(;
    context,
    lowres = true,
)
forcing = ClimaLand.prescribed_forcing_era5(
    era5_ncdata_path,
    domain.space.surface,
    start_date,
    earth_param_set,
    FT;
    max_wind_speed = 25.0,
    time_interpolation_method = LinearInterpolation(PeriodicCalendar()),
    regridder_type = :InterpolationsRegridder,
)

# MODIS LAI is prescribed for the canopy model
println("Loading MODIS LAI data...")
modis_lai_ncdata_path = ClimaLand.Artifacts.modis_lai_multiyear_paths(;
    context,
    start_date,
    end_date = stop_date,
)
LAI = ClimaLand.prescribed_lai_modis(
    modis_lai_ncdata_path,
    domain.space.surface,
    start_date;
    time_interpolation_method = LinearInterpolation(),
)

# Make the model and simulation
println("Creating land model and simulation...")
model = ClimaLand.LandModel{FT}(forcing, LAI, earth_param_set, domain, Δt)
simulation = ClimaLand.Simulations.LandSimulation(
    start_date,
    stop_date,
    Δt,
    model;
    outdir,
)

# Run the simulation
println("Starting simulation...")
@time ClimaLand.Simulations.solve!(simulation)
println("Simulation completed!")

# Let's make some plots!
println("Generating plots...")
LandSimVis.make_annual_timeseries(simulation; savedir = root_path)
LandSimVis.make_heatmaps(simulation; date = stop_date, savedir = root_path)
LandSimVis.make_leaderboard_plots(simulation, savedir = root_path)

println("All plots generated in directory: $root_path")
println("Simulation complete!")
