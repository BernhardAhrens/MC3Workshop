# =============================================================================
# Running a simple soil model
# =============================================================================
# 
# Let's set up and run a simple ClimaLand simulation! This tutorial sets up our 
# EnergyHydrology soil model on a column domain. The model prognoses soil liquid 
# water content, ice water content, and internal energy, so we can keep track of 
# water movement and phase change throughout the soil column.

# =============================================================================
# First we need to set up the Julia environment by adding the packages we'll use.
# =============================================================================

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = 0
required_pkgs = ["Dates", "ClimaParams", "ClimaDiagnostics", "ClimaLand"]
import Pkg
Pkg.Registry.update()
Pkg.add(required_pkgs)

# =============================================================================
# Now we can import the Julia packages to load them into this session.
# =============================================================================

import ClimaParams as CP
using ClimaLand
using ClimaLand.Domains
using ClimaLand.Soil
import ClimaLand.Parameters as LP
import ClimaLand.Simulations: LandSimulation, solve!
import ClimaDiagnostics
using Dates

pkgversion(ClimaLand)

# =============================================================================
# Choose a floating point precision, and get the parameter set, which holds 
# constants used across CliMA models.
# =============================================================================

FT = Float32
earth_param_set = LP.LandParameters(FT);

# =============================================================================
# We will run this simulation on a column domain with 1 meter depth, at a 
# random lat/lon location.
# =============================================================================

zmax = FT(0)
zmin = FT(-1.0)
longlat = FT.((34.1, -118.1))
domain = Domains.Column(; zlim = (zmin, zmax), nelements = 10, longlat);
surface_space = domain.space.surface;

# =============================================================================
# We choose the initial and final simulation times as `DateTime` objects, and 
# a timestep in seconds.
# =============================================================================

start_date = DateTime(2008);
end_date = start_date + Second(60 * 60 * 72);
dt = 1000.0;

# =============================================================================
# The soil model takes in 2 forcing objects, atmosphere and radiation,
# which we read in from ERA5 data.
# =============================================================================

era5_ncdata_path =
    ClimaLand.Artifacts.era5_land_forcing_data2008_path(; lowres = true);
atmos, radiation = ClimaLand.prescribed_forcing_era5(
    era5_ncdata_path,
    surface_space,
    start_date,
    earth_param_set,
    FT,
);

# =============================================================================
# Now, we can create the EnergyHydrology model.
# This constructor uses default parameters and parameterizations, but these can 
# also be overwritten, which we'll demonstrate in later tutorials.
# =============================================================================

model = Soil.EnergyHydrology{FT}(
    domain,
    (; atmos, radiation),
    earth_param_set,
);

# =============================================================================
# We define a function to set initial conditions for the prognostic variables.
# =============================================================================

function set_ic!(Y, p, t0, soil)
    Y.soil.ϑ_l .= FT(0.24);
    Y.soil.θ_i .= FT(0.0);
    T = FT(290.15);
    ρc_s = Soil.volumetric_heat_capacity.(
        Y.soil.ϑ_l,
        Y.soil.θ_i,
        soil.parameters.ρc_ds,
        soil.parameters.earth_param_set,
    );
    Y.soil.ρe_int .=
        Soil.volumetric_internal_energy.(
            Y.soil.θ_i,
            ρc_s,
            T,
            soil.parameters.earth_param_set,
        );
end

# =============================================================================
# Since we'll want to make some plots, let's set up an object to save the 
# model output periodically.
# =============================================================================

diag_writer = ClimaDiagnostics.Writers.DictWriter();
diagnostics = ClimaLand.Diagnostics.default_diagnostics(
    model,
    start_date;
    output_vars = ["swc", "tsoil"],
    output_writer = diag_writer,
    average_period = :hourly,
);

# =============================================================================
# Now construct the `LandSimulation` object, which contains the model
# and additional timestepping information.
# =============================================================================

simulation = LandSimulation(start_date, end_date, dt, model; set_ic!, user_callbacks = (), diagnostics);

# =============================================================================
# Now we can run the simulation!
# =============================================================================

solve!(simulation);

# =============================================================================
# Let's plot some results, for example soil water content and soil temperature 
# over time:
# =============================================================================

# =============================================================================
# First we need to load some more packages. Unfortunately, Julia plotting 
# packages are hefty, so this takes a while.
# =============================================================================

plotting_pkgs = ["CairoMakie", "ClimaAnalysis", "GeoMakie", "Printf", "StatsBase"]
Pkg.add(plotting_pkgs)
using CairoMakie, ClimaAnalysis, GeoMakie, Printf, StatsBase
import ClimaLand.LandSimVis as LandSimVis

LandSimVis.make_timeseries(simulation; short_names = ["swc", "tsoil"]);

# =============================================================================
# The plot will be saved as "variable_timeseries.pdf" in the folder under 
# "Files" to the right; double click the filename to view it.
# 
# Given the very simple experiment we just set up, the plot is also simple. 
# Continue to the other tutorials for more complex simulations and 
# visualizations!
# =============================================================================
