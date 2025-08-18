# =============================================================================
# Changing LandModel Parameterizations
# =============================================================================
# 
# In the previous tutorial, we saw how to set up and run an integrated 
# SoilCanopyModel using default parameters and parameterizations. This tutorial 
# will use another integrated model, the LandModel, which contains soil, canopy, 
# and snow components. Additionally, this time we'll make changes to the 
# parameterizations in use.
# 
# We'll use non-default parameterizations for one canopy component:
# - radiative transfer: change from the default `TwoStreamModel` to `BeerLambertModel`
# 
# and one snow component:
# - snow albedo: change from the default `ConstantAlbedo` to `ZenithAngleAlbedoModel`

# =============================================================================
# Fluxnet simulations with the full land model: snow, soil, canopy
# =============================================================================
# 
# As in the previous tutorial, we'll run at a Fluxnet site - this time the 
# Niwot Ridge site.
# 
# Citation: Peter D. Blanken, Russel K. Monson, Sean P. Burns,
# David R. Bowling, Andrew A. Turnipseed (2022),
# AmeriFlux FLUXNET-1F US-NR1 Niwot Ridge Forest (LTER NWT1),
# Ver. 3-5, AmeriFlux AMP, (Dataset). https://doi.org/10.17190/AMF/1871141

# =============================================================================
# Preliminary Setup
# =============================================================================

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = 0
required_pkgs = ["ClimaParams", "Dates", "ClimaUtilities", "ClimaDiagnostics", "DelimitedFiles", "ClimaLand"]
import Pkg
Pkg.Registry.update()
Pkg.add(required_pkgs)

using Dates
import ClimaParams as CP
using ClimaDiagnostics
using ClimaLand
using ClimaLand.Domains: Column, obtain_surface_domain
using ClimaLand.Simulations
using ClimaLand.Snow
using ClimaLand.Canopy
import ClimaLand.Parameters as LP
using DelimitedFiles
import ClimaLand.FluxnetSimulations as FluxnetSimulations

pkgversion(ClimaLand)

# =============================================================================
# Define the floating point precision desired (64 or 32 bit), and get the
# parameter set holding constants used across CliMA Models.
# =============================================================================

const FT = Float32;
earth_param_set = LP.LandParameters(FT);

# =============================================================================
# We will use prescribed atmospheric and radiative forcing from the
# US-NR1 tower. We also read in the MODIS LAI and let that vary in time 
# in a prescribed manner.
# =============================================================================

site_ID = "US-NR1";
site_ID_val = FluxnetSimulations.replace_hyphen(site_ID);

# =============================================================================
# Get the latitude and longitude in degrees, as well as the
# time offset in hours of local time from UTC
# =============================================================================

(; time_offset, lat, long) =
    FluxnetSimulations.get_location(FT, Val(site_ID_val));

# =============================================================================
# Get the height of the sensors in m
# =============================================================================

(; atmos_h) = FluxnetSimulations.get_fluxtower_height(FT, Val(site_ID_val));

# =============================================================================
# Set a start and stop date of the simulation in UTC, as well as
# a timestep in seconds
# =============================================================================

(start_date, stop_date) =
    FluxnetSimulations.get_data_dates(site_ID, time_offset);
Δt = 450.0;

# =============================================================================
# Setup the domain for the model. This corresponds to
# a column of 2m in depth, with 10 equally spaced layers.
# The lat and long are provided so that we can look up default parameters
# for this location using the default ClimaLand parameter maps.
# =============================================================================

zmin = FT(-2) # in m
zmax = FT(0) # in m
domain = Column(; zlim = (zmin, zmax), nelements = 10, longlat = (long, lat));

# =============================================================================
# Forcing data for the site - this uses our interface for working with Fluxnet data
# =============================================================================

forcing = FluxnetSimulations.prescribed_forcing_fluxnet(
    site_ID,
    lat,
    long,
    time_offset,
    atmos_h,
    start_date,
    earth_param_set,
    FT,
);

# =============================================================================
# LAI for the site - this uses our interface for working with MODIS data.
# =============================================================================

modis_lai_ncdata_path = ClimaLand.Artifacts.modis_lai_multiyear_paths(;
    start_date,
    end_date = stop_date,
);
LAI = ClimaLand.prescribed_lai_modis(
    modis_lai_ncdata_path,
    domain.space.surface,
    start_date,
);

# =============================================================================
# Set up the integrated model
# =============================================================================
# 
# First, we need to set up the component models we won't be using the defaults for.
# Since we are constructing these outside of the `LandModel` constructor,
# we need to provide some additional inputs that were omitted in the previous
# `LandModel` tutorial:
# - `surface_domain`: the surface of this simulation domain, which the snow and canopy models will use
# - `prognostic_land_components`: the prognostic land components, which must be consistent across all components
# - `ground`: the canopy ground conditions, which are prognostic since we're running with the soil

surface_domain = obtain_surface_domain(domain);
prognostic_land_components = (:canopy, :snow, :soil, :soilco2);
ground = ClimaLand.PrognosticGroundConditions{FT}();

# =============================================================================
# First, we will set up the snow model.
# We will use the `ZenithAngleAlbedoModel` for the snow albedo parameterization.
# This parameterization uses the zenith angle of the sun to determine the albedo,
# as opposed to the default `ConstantAlbedoModel` which uses a temporally and
# spatially constant snow albedo.
# =============================================================================

α_0 = FT(0.6) # parameter controlling the minimum snow albedo
Δα = FT(0.06) # parameter controlling the snow albedo when θs = 90∘
k = FT(2) # rate at which albedo drops to its minimum value with zenith angle
α_snow = Snow.ZenithAngleAlbedoModel(α_0, Δα, k);

# =============================================================================
# Now we can create the `SnowModel` model with the specified snow albedo parameterization.
# =============================================================================

snow = Snow.SnowModel(
    FT,
    surface_domain,
    forcing,
    earth_param_set,
    Δt;
    prognostic_land_components,
    α_snow,
);

# =============================================================================
# Now let's set up the canopy model using the `BeerLambertModel` radiative 
# transfer parameterization with custom parameters.
# 
# We'll explore three different ways to construct this parameterization.
# The first way is to use the `BeerLambertModel` constructor with the default parameters.
# This method requires the simulation domain as it reads in radiation
# parameters from a map of CLM parameters by default.
# =============================================================================

radiative_transfer = Canopy.BeerLambertModel{FT}(domain);

# =============================================================================
# Alternatively, we could use the same constructor but provide custom values 
# for a subset of the parameters, and use the global maps for the remainder.
# For example, we might want to use the global maps for albedo, but
# explore how the results change when we assume a spherical
# distribution of leaves (G = 0.5) and no clumping (Ω = 1).
# =============================================================================

G_Function = Canopy.ConstantGFunction(FT(0.5)); # leaf angle distribution value 0.5
Ω = 1; # clumping index
radiation_parameters = (; G_Function, Ω);
radiative_transfer = Canopy.BeerLambertModel{FT}(domain; radiation_parameters);

# =============================================================================
# If you want to overwrite all of the parameters, you do not need to use
# global maps, and therefore do not need the domain.
# In a case like this, we may want to use a different `BeerLambertModel` constructor,
# which takes the parameters object directly:
# =============================================================================

radiative_transfer_parameters = Canopy.BeerLambertParameters(FT; G_Function, Ω);
radiative_transfer = Canopy.BeerLambertModel(radiative_transfer_parameters);

# =============================================================================
# Now that we've explored different ways to set up the `BeerLambertModel`, 
# we can create the `CanopyModel` model with the specified radiative transfer
# parameterization passed as a keyword argument.
# =============================================================================

(; atmos, radiation) = forcing;
canopy = Canopy.CanopyModel{FT}(
    surface_domain,
    (; atmos, radiation, ground),
    LAI,
    earth_param_set;
    prognostic_land_components,
    radiative_transfer,
);

# =============================================================================
# Now we can construct the integrated `LandModel`. Since we want to use the
# defaults for the soil model, we don't need to provide anything for it.
# For the canopy and snow models, we'll provide the models we just set up.
# =============================================================================

land_model =
    LandModel{FT}(forcing, LAI, earth_param_set, domain, Δt; snow, canopy);

# =============================================================================
# Now we define a function to set initial conditions, and set up diagnostics 
# to save model output periodically during the simulation.
# =============================================================================

set_ic! = FluxnetSimulations.make_set_fluxnet_initial_conditions(
    site_ID,
    start_date,
    time_offset,
    land_model,
);
output_vars = ["swu", "lwu", "shf", "lhf", "swe", "swc", "si"]
diagnostics = ClimaLand.default_diagnostics(
    land_model,
    start_date;
    output_writer = ClimaDiagnostics.Writers.DictWriter(),
    output_vars,
    average_period = :hourly,
);

# =============================================================================
# Choose how often we want to update the forcing.
# =============================================================================

data_dt = Second(FluxnetSimulations.get_data_dt(site_ID));
updateat = Array(start_date:data_dt:stop_date);

# =============================================================================
# Now we can construct the simulation object and solve it.
# =============================================================================

simulation = Simulations.LandSimulation(
    start_date,
    stop_date,
    Δt, # seconds
    land_model;
    set_ic!,
    updateat,
    user_callbacks = (),
    diagnostics,
);
solve!(simulation);

# =============================================================================
# Plotting results
# =============================================================================

plotting_pkgs = ["CairoMakie", "ClimaAnalysis", "GeoMakie", "Printf", "StatsBase"]
Pkg.add(plotting_pkgs)
using CairoMakie, ClimaAnalysis, GeoMakie, Printf, StatsBase
import ClimaLand.LandSimVis as LandSimVis

LandSimVis.make_diurnal_timeseries(
    simulation;
    short_names = ["shf", "lhf", "swu", "lwu"],
    spinup_date = start_date + Day(20),
    plot_stem_name = "US_NR1_diurnal_timeseries_parameterizations",
);

LandSimVis.make_timeseries(
    simulation;
    short_names = ["swc", "si", "swe"],
    spinup_date = start_date + Day(20),
    plot_stem_name = "US_NR1_timeseries_parameterizations",
);

# =============================================================================
# The plots will be saved as:
# - swc_US_NR1_timeseries_parameterizations.png
# - si_US_NR1_timeseries_parameterizations.png
# - swe_US_NR1_timeseries_parameterizations.png
# 
# Now you can compare these plots to those generated in the default LandModel 
# Fluxnet tutorial. How are the results different? How are they the same?
# =============================================================================
