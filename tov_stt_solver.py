#!/usr/bin/env python3
"""
TOV Stars with Piecewise Polytropic Equation of State in Scalar-Tensor Theories

This module solves the Tolman-Oppenheimer-Volkoff (TOV) equations for neutron stars
in Scalar-Tensor Theories (STT) of gravity. In STT, gravity is mediated by both 
the metric tensor (as in General Relativity) and a scalar field φ.

The TOV equations describe the hydrostatic equilibrium of a spherically symmetric,
static star. In the Einstein frame, the spacetime metric is:
    ds² = -e^ν dt² + e^λ dr² + r²(dθ² + sin²θ dφ²)

The system of coupled differential equations includes:
- Pressure P(r) and energy density ε(r) evolution
- Mass function m(r) 
- Metric functions ν(r), λ(r)
- Scalar field φ(r) and its radial derivative ψ(r) = dφ/dr

Physics Background:
- Uses a piecewise polytropic equation of state (EOS) to model neutron star matter
- Scalar field coupling affects the stellar structure through the function A(φ) = e^(βφ²/2)
- The parameter β controls the strength of scalar-tensor coupling
- For β = 0, the equations reduce to standard General Relativity

Author: N. Stergioulas (Aristotle University of Thessaloniki)
Converted to Python script with extensive documentation
Version: 1.0 (March 2019)
License: CC BY-NC-SA 4.0 / GNU GPLv3
"""

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate, optimize
from scipy.interpolate import PchipInterpolator
import sys
import warnings
from decimal import Decimal

# Suppress warnings for cleaner output
if not sys.warnoptions:
    warnings.simplefilter("ignore")


class TOVSTTSolver:
    """
    Solver for TOV equations in Scalar-Tensor Theories.
    
    This class implements a complete numerical solver for neutron stars in 
    scalar-tensor theories of gravity, including:
    - Piecewise polytropic equation of state
    - Coupled system of TOV equations
    - Boundary condition optimization
    - Physical property computation
    """
    
    def __init__(self, beta=-4.5):
        """
        Initialize the TOV-STT solver.
        
        Parameters:
        -----------
        beta : float, default=-4.5
            Scalar-tensor coupling parameter. Controls the strength of 
            scalar field interactions. β = 0 corresponds to General Relativity.
            Negative values typical for viable scalar-tensor theories.
        """
        self.beta = beta
        self._setup_physical_constants()
        self._setup_equation_of_state()
        
    def _setup_physical_constants(self):
        """
        Set up fundamental physical constants in CGS units.
        
        The code uses geometrized units (G = c = M_sun = 1) internally,
        with conversion factors stored for output in physical units.
        """
        # Physical constants in CGS units
        self.c = 2.9979e10          # Speed of light [cm/s]
        self.G = 6.67408e-8         # Gravitational constant [cm³/g/s²]
        self.Msun = 1.989e33        # Solar mass [g]
        
        # Geometrized unit conversion factors
        self.Length = self.G * self.Msun / self.c**2    # Length scale [cm]
        self.Time = self.Length / self.c                # Time scale [s]
        self.Density = self.Msun / self.Length**3       # Density scale [g/cm³]
        
        print(f"Geometrized units initialized:")
        print(f"Length scale: {self.Length/1e5:.3f} km")
        print(f"Density scale: {self.Density:.3e} g/cm³")
        
    def _setup_equation_of_state(self):
        """
        Set up the piecewise polytropic equation of state (EOS).
        
        The EOS consists of multiple segments, each described by:
        P = K_i * ρ^Γ_i for different density ranges
        
        This implementation uses the SLy EOS (Douchin & Haensel 2001) for 
        high densities and a low-density crust model.
        
        Physical motivation:
        - Low density: Non-relativistic electrons and nuclei
        - Intermediate: Relativistic electrons, non-relativistic nucleons  
        - High density: Relativistic nucleons, possible exotic matter
        """
        # Transition densities between EOS segments [in geometrized units]
        self.rho1 = pow(10, 14.7) / self.Density     # First high-density transition
        self.rho2 = pow(10, 15.0) / self.Density     # Second high-density transition
        
        # High-density SLy EOS parameters
        p1 = pow(10.0, 34.384) / self.Density / self.c**2  # Transition pressure
        self.Gamma1 = 3.005    # Adiabatic index for intermediate density
        self.Gamma2 = 2.988    # Adiabatic index for high density  
        self.Gamma3 = 2.851    # Adiabatic index for very high density
        
        # Polytropic constants K_i = P_i / ρ_i^Γ_i
        self.K1 = p1 / pow(self.rho1, self.Gamma1)
        self.K2 = self.K1 * pow(self.rho1, self.Gamma1 - self.Gamma2)
        self.K3 = self.K2 * pow(self.rho2, self.Gamma2 - self.Gamma3)
        
        # Low-density crust EOS parameters
        self.rhoL_1 = 2.62789e12 / self.Density   # Nuclear saturation density
        self.rhoL_2 = 3.78358e11 / self.Density   # Neutron drip density
        self.rhoL_3 = 2.44034e7 / self.Density    # Heavy nuclei regime
        self.rhoL_4 = 0.0                         # Surface (vacuum)
        
        # Crust adiabatic indices
        self.GammaL_1 = 1.35692
        self.GammaL_2 = 0.62223  
        self.GammaL_3 = 1.28733
        self.GammaL_4 = 1.58425
        
        # Crust polytropic constants
        self.KL_1 = 3.99874e-8 * pow(self.Msun/self.Length**3, self.GammaL_1-1)
        self.KL_2 = 5.32697e+1 * pow(self.Msun/self.Length**3, self.GammaL_2-1)
        self.KL_3 = 1.06186e-6 * pow(self.Msun/self.Length**3, self.GammaL_3-1)
        self.KL_4 = 6.80110e-9 * pow(self.Msun/self.Length**3, self.GammaL_4-1)
        
        # Compute energy density continuity parameters
        # These ensure ε(ρ) is continuous across EOS boundaries
        self._compute_energy_continuity()
        
        # Store all EOS parameters for easy access
        self._package_eos_parameters()
        
        print(f"Piecewise polytropic EOS initialized with {len(self._get_density_ranges())} segments")
        
    def _compute_energy_continuity(self):
        """
        Compute energy density continuity parameters α_i.
        
        The energy density is given by:
        ε = (1 + α_i)ρ + K_i/(Γ_i - 1) * ρ^Γ_i
        
        The α_i parameters ensure continuity: ε_i(ρ_i) = ε_{i+1}(ρ_i)
        This is essential for thermodynamic consistency.
        """
        # Start from the surface and work inward
        self.epsL_4 = 0.0
        self.alphaL_4 = 0.0
        
        # Each α_i is determined by requiring energy density continuity
        self.epsL_3 = ((1 + self.alphaL_4) * self.rhoL_3 + 
                       self.KL_4/(self.GammaL_4 - 1) * pow(self.rhoL_3, self.GammaL_4))
        self.alphaL_3 = (self.epsL_3/self.rhoL_3 - 1 - 
                         self.KL_3/(self.GammaL_3 - 1) * pow(self.rhoL_3, self.GammaL_3 - 1))
        
        self.epsL_2 = ((1 + self.alphaL_3) * self.rhoL_2 + 
                       self.KL_3/(self.GammaL_3 - 1) * pow(self.rhoL_2, self.GammaL_3))
        self.alphaL_2 = (self.epsL_2/self.rhoL_2 - 1 - 
                         self.KL_2/(self.GammaL_2 - 1) * pow(self.rhoL_2, self.GammaL_2 - 1))
        
        self.epsL_1 = ((1 + self.alphaL_2) * self.rhoL_1 + 
                       self.KL_2/(self.GammaL_2 - 1) * pow(self.rhoL_1, self.GammaL_2))
        self.alphaL_1 = (self.epsL_1/self.rhoL_1 - 1 - 
                         self.KL_1/(self.GammaL_1 - 1) * pow(self.rhoL_1, self.GammaL_1 - 1))
        
        # Match crust to high-density EOS
        self.rho0 = pow(self.KL_1/self.K1, 1.0/(self.Gamma1 - self.GammaL_1))
        self.eps0 = ((1.0 + self.alphaL_1) * self.rho0 + 
                     self.KL_1/(self.GammaL_1 - 1.0) * pow(self.rho0, self.GammaL_1))
        
        # High-density continuity parameters
        self.alpha1 = (self.eps0/self.rho0 - 1 - 
                       self.K1/(self.Gamma1 - 1) * pow(self.rho0, self.Gamma1 - 1))
        self.eps1 = ((1 + self.alpha1) * self.rho1 + 
                     self.K1/(self.Gamma1 - 1) * pow(self.rho1, self.Gamma1))
        self.alpha2 = (self.eps1/self.rho1 - 1 - 
                       self.K2/(self.Gamma2 - 1) * pow(self.rho1, self.Gamma2 - 1))
        self.eps2 = ((1 + self.alpha2) * self.rho2 + 
                     self.K2/(self.Gamma2 - 1) * pow(self.rho2, self.Gamma2))
        self.alpha3 = (self.eps2/self.rho2 - 1 - 
                       self.K3/(self.Gamma3 - 1) * pow(self.rho2, self.Gamma3 - 1))
        
        # Store transition pressures for EOS evaluation
        self.pL_3 = self.KL_3 * pow(self.rhoL_3, self.GammaL_3)
        self.pL_2 = self.KL_2 * pow(self.rhoL_2, self.GammaL_2)
        self.pL_1 = self.KL_1 * pow(self.rhoL_1, self.GammaL_1)
        self.p0 = self.KL_1 * pow(self.rho0, self.GammaL_1)
        self.p1 = self.K1 * pow(self.rho1, self.Gamma1)
        self.p2 = self.K2 * pow(self.rho2, self.Gamma2)
        
    def _package_eos_parameters(self):
        """Package EOS parameters into tuples for function calls."""
        self.args = (self.rhoL_3, self.rhoL_2, self.rhoL_1, self.rho0, self.rho1, self.rho2,
                     self.KL_4, self.KL_3, self.KL_2, self.KL_1, self.K1, self.K2, self.K3,
                     self.GammaL_4, self.GammaL_3, self.GammaL_2, self.GammaL_1, 
                     self.Gamma1, self.Gamma2, self.Gamma3)
        
        self.args2 = (self.rhoL_3, self.rhoL_2, self.rhoL_1, self.rho0, self.rho1, self.rho2,
                      self.KL_4, self.KL_3, self.KL_2, self.KL_1, self.K1, self.K2, self.K3,
                      self.GammaL_4, self.GammaL_3, self.GammaL_2, self.GammaL_1, 
                      self.Gamma1, self.Gamma2, self.Gamma3,
                      self.pL_3, self.pL_2, self.pL_1, self.p0, self.p1, self.p2,
                      self.alphaL_4, self.alphaL_3, self.alphaL_2, self.alphaL_1, 
                      self.alpha1, self.alpha2, self.alpha3)
    
    def _get_density_ranges(self):
        """Return density ranges for EOS segments for diagnostics."""
        return [
            (0, self.rhoL_3, "Surface to heavy nuclei"),
            (self.rhoL_3, self.rhoL_2, "Heavy nuclei regime"),
            (self.rhoL_2, self.rhoL_1, "Neutron drip region"),
            (self.rhoL_1, self.rho0, "Nuclear saturation"),
            (self.rho0, self.rho1, "Low supranuclear"),
            (self.rho1, self.rho2, "Intermediate supranuclear"),
            (self.rho2, np.inf, "High supranuclear")
        ]
    
    def P_of_rho(self, rho):
        """
        Compute pressure from density using piecewise polytropic EOS.
        
        Parameters:
        -----------
        rho : float or array
            Rest mass density in geometrized units
            
        Returns:
        --------
        float or array
            Pressure in geometrized units
            
        Physics:
        --------
        Each segment follows P = K_i * ρ^Γ_i where:
        - K_i: polytropic constant (has units of pressure/density^Γ_i)
        - Γ_i: adiabatic index (dimensionless)
        - Transition points chosen to match realistic neutron star EOS
        """
        if rho < self.rhoL_3:
            return self.KL_4 * pow(rho, self.GammaL_4)
        elif self.rhoL_3 <= rho < self.rhoL_2:
            return self.KL_3 * pow(rho, self.GammaL_3)
        elif self.rhoL_2 <= rho < self.rhoL_1:
            return self.KL_2 * pow(rho, self.GammaL_2)
        elif self.rhoL_1 <= rho < self.rho0:
            return self.KL_1 * pow(rho, self.GammaL_1)
        elif self.rho0 <= rho < self.rho1:
            return self.K1 * pow(rho, self.Gamma1)
        elif self.rho1 <= rho < self.rho2:
            return self.K2 * pow(rho, self.Gamma2)
        else:
            return self.K3 * pow(rho, self.Gamma3)
    
    def rho_of_P(self, p):
        """
        Compute density from pressure (inverse of P_of_rho).
        
        Parameters:
        -----------
        p : float or array
            Pressure in geometrized units
            
        Returns:
        --------
        float or array
            Rest mass density in geometrized units
            
        Physics:
        --------
        Inverts the polytropic relation: ρ = (P/K_i)^(1/Γ_i)
        """
        if p < self.pL_3:
            return pow(p/self.KL_4, 1.0/self.GammaL_4)
        elif self.pL_3 <= p < self.pL_2:
            return pow(p/self.KL_3, 1.0/self.GammaL_3)
        elif self.pL_2 <= p < self.pL_1:
            return pow(p/self.KL_2, 1.0/self.GammaL_2)
        elif self.pL_1 <= p < self.p0:
            return pow(p/self.KL_1, 1.0/self.GammaL_1)
        elif self.p0 <= p < self.p1:
            return pow(p/self.K1, 1.0/self.Gamma1)
        elif self.p1 <= p < self.p2:
            return pow(p/self.K2, 1.0/self.Gamma2)
        else:
            return pow(p/self.K3, 1.0/self.Gamma3)
    
    def eps_of_rho(self, rho):
        """
        Compute energy density from rest mass density.
        
        Parameters:
        -----------
        rho : float or array
            Rest mass density in geometrized units
            
        Returns:
        --------
        float or array
            Energy density in geometrized units
            
        Physics:
        --------
        Energy density includes rest mass energy plus internal energy:
        ε = (1 + α_i)ρ + K_i/(Γ_i - 1) * ρ^Γ_i
        
        The first term is rest mass + binding energy contribution.
        The second term is the relativistic internal energy from pressure.
        """
        if rho < self.rhoL_3:
            return ((1.0 + self.alphaL_4) * rho + 
                    self.KL_4/(self.GammaL_4 - 1.0) * pow(rho, self.GammaL_4))
        elif self.rhoL_3 <= rho < self.rhoL_2:
            return ((1.0 + self.alphaL_3) * rho + 
                    self.KL_3/(self.GammaL_3 - 1.0) * pow(rho, self.GammaL_3))
        elif self.rhoL_2 <= rho < self.rhoL_1:
            return ((1.0 + self.alphaL_2) * rho + 
                    self.KL_2/(self.GammaL_2 - 1.0) * pow(rho, self.GammaL_2))
        elif self.rhoL_1 <= rho < self.rho0:
            return ((1.0 + self.alphaL_1) * rho + 
                    self.KL_1/(self.GammaL_1 - 1.0) * pow(rho, self.GammaL_1))
        elif self.rho0 <= rho < self.rho1:
            return ((1.0 + self.alpha1) * rho + 
                    self.K1/(self.Gamma1 - 1.0) * pow(rho, self.Gamma1))
        elif self.rho1 <= rho < self.rho2:
            return ((1.0 + self.alpha2) * rho + 
                    self.K2/(self.Gamma2 - 1.0) * pow(rho, self.Gamma2))
        else:
            return ((1.0 + self.alpha3) * rho + 
                    self.K3/(self.Gamma3 - 1.0) * pow(rho, self.Gamma3))
    
    def eps_of_P(self, p):
        """
        Compute energy density from pressure.
        
        Parameters:
        -----------
        p : float or array
            Pressure in geometrized units
            
        Returns:
        --------
        float or array
            Energy density in geometrized units
            
        Physics:
        --------
        Combines density-pressure and density-energy relations:
        ε(P) = ε(ρ(P))
        """
        if p < self.pL_3:
            return ((1.0 + self.alphaL_4) * pow(p/self.KL_4, 1.0/self.GammaL_4) + 
                    p/(self.GammaL_4 - 1))
        elif self.pL_3 <= p < self.pL_2:
            return ((1.0 + self.alphaL_3) * pow(p/self.KL_3, 1.0/self.GammaL_3) + 
                    p/(self.GammaL_3 - 1))
        elif self.pL_2 <= p < self.pL_1:
            return ((1.0 + self.alphaL_2) * pow(p/self.KL_2, 1.0/self.GammaL_2) + 
                    p/(self.GammaL_2 - 1))
        elif self.pL_1 <= p < self.p0:
            return ((1.0 + self.alphaL_1) * pow(p/self.KL_1, 1.0/self.GammaL_1) + 
                    p/(self.GammaL_1 - 1))
        elif self.p0 <= p < self.p1:
            return ((1.0 + self.alpha1) * pow(p/self.K1, 1.0/self.Gamma1) + 
                    p/(self.Gamma1 - 1))
        elif self.p1 <= p < self.p2:
            return ((1.0 + self.alpha2) * pow(p/self.K2, 1.0/self.Gamma2) + 
                    p/(self.Gamma2 - 1))
        else:
            return ((1.0 + self.alpha3) * pow(p/self.K3, 1.0/self.Gamma3) + 
                    p/(self.Gamma3 - 1))
    
    def A_of_phi(self, phi):
        """
        Compute the scalar field coupling function A(φ).
        
        Parameters:
        -----------
        phi : float or array
            Scalar field value
            
        Returns:
        --------
        float or array
            Coupling function A(φ) = exp(βφ²/2)
            
        Physics:
        --------
        This function controls how the scalar field couples to matter.
        - A(φ) = 1 + O(φ²) for weak fields
        - The exponential form ensures A(φ) > 0 always
        - β < 0 typical for viable scalar-tensor theories
        """
        return np.exp(0.5 * self.beta * phi**2)
    
    def alpha_of_phi(self, phi):
        """
        Compute the scalar field coupling derivative α(φ) = dA/dφ / A.
        
        Parameters:
        -----------
        phi : float or array
            Scalar field value
            
        Returns:
        --------
        float or array
            Coupling derivative α(φ) = βφ
            
        Physics:
        --------
        This appears in the TOV equations as the coupling between
        scalar field gradients and matter. It determines how strongly
        the scalar field affects stellar structure.
        """
        return self.beta * phi
    
    def tov_interior(self, r, y):
        """
        TOV system of equations for the stellar interior.
        
        Parameters:
        -----------
        r : float
            Radial coordinate in geometrized units
        y : array-like, shape (5,)
            State vector [P(r), m(r), ν(r), φ(r), ψ(r)]
            
        Returns:
        --------
        array, shape (5,)
            Derivatives [dP/dr, dm/dr, dν/dr, dφ/dr, dψ/dr]
            
        Physics:
        --------
        The TOV equations in scalar-tensor theory:
        
        1. Hydrostatic equilibrium (modified by scalar field):
           dP/dr = -(ε+P)[gravitational + scalar contributions]
           
        2. Mass continuity (includes scalar field energy):
           dm/dr = 4πr²ε + scalar field energy density
           
        3. Metric component evolution:
           dν/dr = 2(m + 4πr³P)/(r(r-2m)) + scalar contributions
           
        4. Scalar field evolution:
           dφ/dr = ψ(r)
           
        5. Scalar field derivative evolution:
           dψ/dr = source terms from matter coupling + geometric terms
        """
        P, m, nu, phi, psi = y
        
        # Compute matter properties
        eps = self.eps_of_P(P)
        A = self.A_of_phi(phi)
        alpha = self.alpha_of_phi(phi)
        
        # TOV equations with scalar field modifications
        dP_dr = (-(eps + P) * 
                 ((m + 4.0*np.pi*A**4*r**3*P) / (r*(r - 2.0*m)) + 
                  0.5*r*psi**2 + alpha*psi))
        
        dm_dr = (4*np.pi*A**4*r**2*eps + 
                 0.5*r*(r - 2*m)*psi**2)
        
        dnu_dr = (2.0*(m + 4.0*np.pi*A**4*r**3*P) / (r*(r - 2.0*m)) + 
                  r*psi**2)
        
        dphi_dr = psi
        
        dpsi_dr = (4*np.pi*A**4*r/(r - 2.0*m) * 
                   (alpha*(eps - 3.0*P) + r*(eps - P)*psi) - 
                   2*(r - m)*psi/(r*(r - 2.0*m)))
        
        return [dP_dr, dm_dr, dnu_dr, dphi_dr, dpsi_dr]
    
    def tov_exterior(self, r, y):
        """
        TOV system of equations for the exterior vacuum region.
        
        Parameters:
        -----------
        r : float
            Radial coordinate in geometrized units
        y : array-like, shape (5,)
            State vector [P(r), m(r), ν(r), φ(r), ψ(r)]
            
        Returns:
        --------
        array, shape (5,)
            Derivatives [dP/dr, dm/dr, dν/dr, dφ/dr, dψ/dr]
            
        Physics:
        --------
        In the exterior vacuum region:
        - Pressure P = 0 (no matter)
        - Mass m(r) changes only due to scalar field energy
        - Metric and scalar field evolve according to vacuum field equations
        - Solutions should approach Schwarzschild + scalar hair asymptotically
        """
        P, m, nu, phi, psi = y
        
        # Vacuum TOV equations
        dP_dr = 0  # No pressure in vacuum
        
        dm_dr = 0.5*r*(r - 2*m)*psi**2  # Only scalar field energy contributes
        
        dnu_dr = 2.0*m/(r*(r - 2.0*m)) + r*psi**2
        
        dphi_dr = psi
        
        dpsi_dr = -2*(r - m)*psi/(r*(r - 2.0*m))
        
        return [dP_dr, dm_dr, dnu_dr, dphi_dr, dpsi_dr]
    
    def solve_star(self, rho_c, initial_guess=(-1.0, -0.106), N=32001):
        """
        Solve for a complete stellar structure given central density.
        
        Parameters:
        -----------
        rho_c : float
            Central density in geometrized units
        initial_guess : tuple, default=(-1.0, -0.106)
            Initial guess for (ν_c, φ_c) optimization
        N : int, default=32001
            Number of radial grid points
            
        Returns:
        --------
        dict
            Complete solution containing:
            - Physical properties (mass, radius, etc.)
            - Full radial profiles
            - Boundary conditions used
            
        Physics:
        --------
        The boundary value problem requires:
        1. Central conditions: P(0)=P_c, m(0)=0, ψ(0)=0
        2. Asymptotic conditions: ν(∞)=0, φ(∞)=0
        3. Optimization to find ν_c, φ_c satisfying these conditions
        """
        # Set central conditions
        self.rho_c = rho_c
        self.eps_c = self.eps_of_rho(rho_c)
        self.P_c = self.P_of_rho(rho_c) 
        
        print(f"Solving star with central density: {rho_c*self.Density:.3e} g/cm³")
        print(f"Central pressure: {self.P_c*self.Density*self.c**2:.3e} dyn/cm²")
        
        # Set up radial grid
        # Grid extends to several times the expected stellar radius
        r_max = 128*4.0 * pow(3.0/(4.0*np.pi*self.eps_c), 1.0/3.0)
        self.r = np.linspace(0.0, r_max, N)
        self.dr = self.r[1] - self.r[0]
        
        print(f"Radial grid: {N} points, r_max = {r_max*self.Length/1e5:.1f} km")
        
        # Optimize boundary conditions
        print("Optimizing boundary conditions...")
        result = optimize.minimize(self._trial_solution, initial_guess, 
                                 method='nelder-mead',
                                 options={'disp': False})
        
        if not result.success:
            print(f"Warning: Optimization may not have converged: {result.message}")
        
        # Generate final solution with optimal parameters
        nu_c, phi_c = result.x
        print(f"Optimal boundary conditions: ν_c = {nu_c:.6f}, φ_c = {phi_c:.6f}")
        
        return self._generate_final_solution(nu_c, phi_c)
    
    def _trial_solution(self, params):
        """
        Trial solution for boundary condition optimization.
        
        Parameters:
        -----------
        params : tuple
            Trial values for (ν_c, φ_c)
            
        Returns:
        --------
        float
            Objective function value (should be minimized to zero)
            
        Physics:
        --------
        The objective is to find ν_c, φ_c such that ν(∞) → 0 and φ(∞) → 0.
        We minimize |ν(r_max)|² + |φ(r_max)|² as a proxy for the asymptotic
        conditions.
        """
        nu_c, phi_c = params
        
        # Initial conditions
        y0 = [self.P_c, 0.0, nu_c, phi_c, 0.0]
        
        # Integrate interior solution
        solver = integrate.ode(self.tov_interior)
        solver.set_integrator('lsoda', rtol=1e-12, atol=1e-50)
        solver.set_initial_value(y0, self.dr)
        
        # Find stellar surface (where pressure drops to zero)
        idx = 1
        while solver.successful() and solver.t < self.r[-1] and solver.y[0] > 0.0:
            solver.integrate(solver.t + self.dr)
            idx += 1
        
        surface_idx = idx - 1
        surface_conditions = solver.y
        
        # Continue with exterior integration
        solver_ext = integrate.ode(self.tov_exterior)
        solver_ext.set_integrator('lsoda', rtol=1e-12, atol=1e-50)
        solver_ext.set_initial_value(surface_conditions, self.r[surface_idx])
        
        # Integrate to grid boundary
        current_idx = surface_idx
        while solver_ext.successful() and solver_ext.t < self.r[-1]:
            solver_ext.integrate(solver_ext.t + self.dr)
            current_idx += 1
        
        # Objective: minimize deviation from asymptotic conditions
        final_nu = solver_ext.y[2]
        final_phi = solver_ext.y[3]
        objective = np.sqrt(final_nu**2 + final_phi**2)
        
        return objective
    
    def _generate_final_solution(self, nu_c, phi_c):
        """
        Generate the complete stellar solution with optimized boundary conditions.
        
        Parameters:
        -----------
        nu_c, phi_c : float
            Optimized central values for metric and scalar field
            
        Returns:
        --------
        dict
            Complete solution dictionary
        """
        # Initial conditions
        y0 = [self.P_c, 0.0, nu_c, phi_c, 0.0]
        
        # Storage for solution
        N = len(self.r)
        solution = np.zeros((N, 5))
        solution[0, :] = y0
        
        # Integrate interior
        solver = integrate.ode(self.tov_interior)
        solver.set_integrator('lsoda', rtol=1e-12, atol=1e-50)
        solver.set_initial_value(y0, self.dr)
        
        idx = 1
        while solver.successful() and solver.t < self.r[-1] and solver.y[0] > 0.0:
            solution[idx, :] = solver.y
            solver.integrate(solver.t + self.dr)
            idx += 1
        
        surface_idx = idx - 1
        radius = self.r[surface_idx]
        mass_interior = solution[surface_idx, 1]
        
        print(f"Stellar surface found at r = {radius*self.Length/1e5:.3f} km")
        print(f"Interior mass: {mass_interior:.3f} M_sun")
        
        # Integrate exterior
        solver_ext = integrate.ode(self.tov_exterior)
        solver_ext.set_integrator('lsoda', rtol=1e-12, atol=1e-50)
        solver_ext.set_initial_value(solution[surface_idx, :], self.r[surface_idx])
        
        current_idx = surface_idx
        while solver_ext.successful() and solver_ext.t < self.r[-1]:
            solution[current_idx, :] = solver_ext.y
            solver_ext.integrate(solver_ext.t + self.dr)
            current_idx += 1
        
        # Compute derived quantities
        radius_accurate = self._find_accurate_radius(solution, surface_idx)
        mass_asymptotic = self._compute_asymptotic_mass(solution)
        scalar_charge = self._compute_scalar_charge(solution)
        
        print(f"Accurate radius: {radius_accurate*self.Length/1e5:.3f} km")
        print(f"Asymptotic mass: {mass_asymptotic:.6f} M_sun")
        print(f"Scalar charge: {scalar_charge*self.Length/1e5:.3f} km")
        
        # Package results
        return {
            'r': self.r,
            'solution': solution,
            'surface_idx': surface_idx,
            'radius': radius_accurate,
            'mass': mass_asymptotic,
            'scalar_charge': scalar_charge,
            'central_density': self.rho_c,
            'central_pressure': self.P_c,
            'central_energy_density': self.eps_c,
            'nu_c': nu_c,
            'phi_c': phi_c,
            'beta': self.beta
        }
    
    def _find_accurate_radius(self, solution, surface_idx):
        """
        Find stellar radius more accurately using interpolation.
        
        The radius is defined where the specific enthalpy h = (ε+P)/ρ - 1 = 0.
        This is a more robust condition than P = 0 for numerical purposes.
        """
        # Extract data near the surface
        r_data = self.r[surface_idx-3:surface_idx+1]
        h_data = np.zeros(4)
        
        for i in range(4):
            idx = surface_idx - 3 + i
            P = solution[idx, 0]
            if P > 0:
                rho = self.rho_of_P(P)
                eps = self.eps_of_P(P)
                h_data[i] = (eps + P)/rho - 1.0
            else:
                h_data[i] = 0.0
        
        # Interpolate and find zero
        h_interp = PchipInterpolator(r_data, h_data)
        radius = optimize.brentq(h_interp, r_data[0], r_data[-1], xtol=1e-16)
        
        return radius
    
    def _compute_asymptotic_mass(self, solution):
        """
        Compute the asymptotic gravitational mass from metric behavior.
        
        At large r, the metric component ν(r) behaves as:
        ν(r) ≈ ln(1 - 2M/r) + O(1/r²)
        
        So M can be extracted from: M = r²(dν/dr)e^ν/2
        """
        # Use values near the outer boundary
        idx = -10  # Use point near boundary but not exactly at boundary
        r_val = self.r[idx]
        nu_val = solution[idx, 2]
        
        # Compute derivative numerically
        dnu_dr = np.gradient(solution[:, 2], self.dr)[idx]
        
        mass = dnu_dr * r_val**2 * np.exp(nu_val) / 2.0
        
        return mass
    
    def _compute_scalar_charge(self, solution):
        """
        Compute the scalar charge from asymptotic scalar field behavior.
        
        At large r, the scalar field behaves as:
        φ(r) ≈ ω/r + O(1/r²)
        
        The scalar charge is ω = -r²(dφ/dr)|_{r→∞}
        """
        # Use values near the outer boundary
        idx = -10
        r_val = self.r[idx]
        
        # Compute derivative numerically
        dphi_dr = np.gradient(solution[:, 3], self.dr)[idx]
        
        scalar_charge = -dphi_dr * r_val**2
        
        return scalar_charge
    
    def save_solution(self, result, filename_base="tov_stt"):
        """
        Save the stellar solution to data files.
        
        Parameters:
        -----------
        result : dict
            Solution dictionary from solve_star()
        filename_base : str
            Base filename for output files
        """
        N = len(result['r'])
        
        # Prepare data arrays in geometrized units
        data_geom = np.zeros((N, 11))
        data_geom[:, 0] = result['r']  # radius
        
        for i in range(N):
            P = result['solution'][i, 0]
            data_geom[i, 1] = self.rho_of_P(P) if P > 0 else 0  # density
            data_geom[i, 2] = self.eps_of_P(P) if P > 0 else 0  # energy density
            data_geom[i, 3] = P  # pressure
            data_geom[i, 4] = result['solution'][i, 1]  # mass function
            data_geom[i, 5] = result['solution'][i, 2]  # nu
            
            # Lambda metric component
            if result['r'][i] > 0:
                data_geom[i, 6] = -np.log(1.0 - 2.0*result['solution'][i, 1]/result['r'][i])
            else:
                data_geom[i, 6] = 0.0
                
            # Specific enthalpy
            if P > 0:
                rho = data_geom[i, 1] 
                eps = data_geom[i, 2]
                data_geom[i, 7] = (eps + P)/rho - 1.0
            else:
                data_geom[i, 7] = 0.0
                
            data_geom[i, 8] = result['solution'][i, 3]  # scalar field phi
            data_geom[i, 9] = result['solution'][i, 4]  # scalar field derivative psi
            
            # Local adiabatic index
            rho = data_geom[i, 1]
            if rho < self.rhoL_3:
                data_geom[i, 10] = self.GammaL_4
            elif self.rhoL_3 <= rho < self.rhoL_2:
                data_geom[i, 10] = self.GammaL_3
            elif self.rhoL_2 <= rho < self.rhoL_1:
                data_geom[i, 10] = self.GammaL_2
            elif self.rhoL_1 <= rho < self.rho0:
                data_geom[i, 10] = self.GammaL_1
            elif self.rho0 <= rho < self.rho1:
                data_geom[i, 10] = self.Gamma1
            elif self.rho1 <= rho < self.rho2:
                data_geom[i, 10] = self.Gamma2
            else:
                data_geom[i, 10] = self.Gamma3
        
        # Convert to CGS units
        data_cgs = np.zeros((N, 10))
        data_cgs[:, 0] = data_geom[:, 0] * self.Length  # radius [cm]
        data_cgs[:, 1] = data_geom[:, 1] * self.Density  # density [g/cm³]
        data_cgs[:, 2] = data_geom[:, 2] * self.Density * self.c**2  # energy density [erg/cm³]
        data_cgs[:, 3] = data_geom[:, 3] * self.Density * self.c**2  # pressure [dyn/cm²]
        data_cgs[:, 4] = data_geom[:, 4] * self.Msun  # mass [g]
        data_cgs[:, 5] = data_geom[:, 5]  # nu [dimensionless]
        data_cgs[:, 6] = data_geom[:, 6]  # lambda [dimensionless]
        data_cgs[:, 7] = data_geom[:, 7] * self.c**2  # specific enthalpy [cm²/s²]
        data_cgs[:, 8] = data_geom[:, 8]  # scalar field [dimensionless]
        data_cgs[:, 9] = data_geom[:, 10]  # adiabatic index [dimensionless]
        
        # Save files
        header_geom = ("Columns: r, rho, epsilon, P, m, nu, lambda, h, phi, psi, Gamma\n"
                      "Units: Geometrized (G=c=M_sun=1)")
        header_cgs = ("Columns: r[cm], rho[g/cm³], eps[erg/cm³], P[dyn/cm²], m[g], "
                     "nu, lambda, h[cm²/s²], phi, Gamma\n"
                     "Physical (CGS) units")
        
        np.savetxt(f"{filename_base}.dat", data_geom, 
                  header=header_geom, fmt='%.12e')
        np.savetxt(f"{filename_base}_cgs.dat", data_cgs, 
                  header=header_cgs, fmt='%.12e')
        
        print(f"Solution saved to {filename_base}.dat and {filename_base}_cgs.dat")
    
    def plot_solution(self, result, show_plots=True, save_plots=False, filename_base="tov_stt"):
        """
        Create comprehensive plots of the stellar solution.
        
        Parameters:
        -----------
        result : dict
            Solution dictionary from solve_star()
        show_plots : bool
            Whether to display plots
        save_plots : bool
            Whether to save plots to files
        filename_base : str
            Base filename for saved plots
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Convert radius to km for plotting
        r_km = result['r'] * self.Length / 1e5
        surface_km = result['radius'] * self.Length / 1e5
        
        # Plot 1: Pressure
        ax1 = plt.subplot(2, 3, 1)
        P_cgs = np.array([self.P_of_rho(self.rho_of_P(result['solution'][i, 0])) 
                         if result['solution'][i, 0] > 0 else 0 
                         for i in range(len(result['r']))]) * self.Density * self.c**2
        mask = P_cgs > 0
        plt.loglog(r_km[mask], P_cgs[mask], 'b-', linewidth=2)
        plt.axvline(surface_km, color='r', linestyle='--', alpha=0.7, label='Surface')
        plt.xlabel('Radius [km]')
        plt.ylabel('Pressure [dyn/cm²]')
        plt.title('Pressure Profile')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 2: Energy density
        ax2 = plt.subplot(2, 3, 2)
        eps_cgs = np.array([self.eps_of_P(result['solution'][i, 0]) 
                           if result['solution'][i, 0] > 0 else 0 
                           for i in range(len(result['r']))]) * self.Density * self.c**2
        mask = eps_cgs > 0
        plt.loglog(r_km[mask], eps_cgs[mask]/self.c**2, 'r-', linewidth=2)
        plt.axvline(surface_km, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Radius [km]')
        plt.ylabel('Energy Density [g/cm³]')
        plt.title('Energy Density Profile')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Mass function
        ax3 = plt.subplot(2, 3, 3)
        mass_msun = result['solution'][:, 1]
        plt.plot(r_km, mass_msun, 'g-', linewidth=2)
        plt.axvline(surface_km, color='r', linestyle='--', alpha=0.7)
        plt.axhline(result['mass'], color='k', linestyle=':', alpha=0.7, 
                   label=f'Total Mass = {result["mass"]:.3f} M☉')
        plt.xlabel('Radius [km]')
        plt.ylabel('Enclosed Mass [M☉]')
        plt.title('Mass Function')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 4: Metric function ν
        ax4 = plt.subplot(2, 3, 4)
        plt.plot(r_km, result['solution'][:, 2], 'c-', linewidth=2)
        plt.axvline(surface_km, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Radius [km]')
        plt.ylabel('ν(r)')
        plt.title('Metric Function ν')
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Scalar field
        ax5 = plt.subplot(2, 3, 5)
        plt.plot(r_km, result['solution'][:, 3], 'm-', linewidth=2, label='φ(r)')
        plt.axvline(surface_km, color='r', linestyle='--', alpha=0.7)
        plt.axhline(0, color='k', linestyle=':', alpha=0.5)
        plt.xlabel('Radius [km]')
        plt.ylabel('Scalar Field φ')
        plt.title(f'Scalar Field (β = {self.beta})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 6: Scalar field derivative
        ax6 = plt.subplot(2, 3, 6)
        plt.plot(r_km, result['solution'][:, 4], 'orange', linewidth=2, label='ψ(r)')
        plt.axvline(surface_km, color='r', linestyle='--', alpha=0.7)
        plt.axhline(0, color='k', linestyle=':', alpha=0.5)
        plt.xlabel('Radius [km]')
        plt.ylabel('ψ = dφ/dr')
        plt.title('Scalar Field Derivative')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{filename_base}_profiles.png", dpi=300, bbox_inches='tight')
            print(f"Profile plots saved to {filename_base}_profiles.png")
            
        if show_plots:
            plt.show()
    
    def print_summary(self, result):
        """
        Print a comprehensive summary of the stellar solution.
        
        Parameters:
        -----------
        result : dict
            Solution dictionary from solve_star()
        """
        print("\n" + "="*60)
        print("         TOV STAR SOLUTION SUMMARY")
        print("="*60)
        
        print(f"\nScalar-Tensor Theory Parameters:")
        print(f"  β (coupling parameter):     {self.beta:8.3f}")
        print(f"  A(φ) coupling function:     exp(βφ²/2)")
        
        print(f"\nCentral Conditions:")
        print(f"  Central density:            {result['central_density']*self.Density:.3e} g/cm³")
        print(f"  Central pressure:           {result['central_pressure']*self.Density*self.c**2:.3e} dyn/cm²")
        print(f"  Central energy density:     {result['central_energy_density']*self.Density:.3e} g/cm³")
        print(f"  Central ν value:            {result['nu_c']:8.6f}")
        print(f"  Central φ value:            {result['phi_c']:8.6f}")
        
        print(f"\nStellar Properties:")
        print(f"  Radius:                     {result['radius']*self.Length/1e5:8.3f} km")
        print(f"  Gravitational mass:         {result['mass']:8.6f} M☉")
        print(f"  Gravitational mass:         {result['mass']*self.Msun:.3e} g")
        print(f"  Scalar charge:              {result['scalar_charge']*self.Length/1e5:8.3f} km")
        print(f"  Compactness (2M/R):         {2*result['mass']/result['radius']:8.6f}")
        
        # Surface conditions
        surface_idx = result['surface_idx']
        phi_surface = result['solution'][surface_idx, 3]
        print(f"\nSurface Conditions:")
        print(f"  Surface φ value:            {phi_surface:8.6f}")
        print(f"  φ change (center to surface): {result['phi_c'] - phi_surface:8.6f}")
        
        # Equation of state info
        print(f"\nEquation of State:")
        print(f"  Type:                       Piecewise polytropic")
        print(f"  High-density model:         SLy EOS")
        print(f"  Number of EOS segments:     {len(self._get_density_ranges())}")
        
        # Numerical parameters
        N = len(result['r'])
        r_max_km = result['r'][-1] * self.Length / 1e5
        print(f"\nNumerical Parameters:")
        print(f"  Grid points:                {N}")
        print(f"  Maximum radius:             {r_max_km:.1f} km")
        print(f"  Grid spacing:               {self.dr*self.Length:.1f} cm")
        
        print("="*60)


def main():
    """
    Main function demonstrating the TOV-STT solver usage.
    
    This example solves for a neutron star with specified central density
    in a scalar-tensor theory with β = -4.5.
    """
    print("TOV Neutron Star Solver for Scalar-Tensor Theories")
    print("="*55)
    
    # Initialize solver with scalar-tensor coupling
    beta = -4.5  # Typical value for viable scalar-tensor theories
    solver = TOVSTTSolver(beta=beta)
    
    # Set central density (in CGS units, then convert to geometrized)
    rho_c_cgs = 1.128e15  # g/cm³ - typical neutron star central density
    rho_c = rho_c_cgs / solver.Density
    
    print(f"\nSolving for neutron star with:")
    print(f"  Central density: {rho_c_cgs:.3e} g/cm³")
    print(f"  ST coupling β:   {beta}")
    
    # Solve the stellar structure
    result = solver.solve_star(rho_c, N=32001)
    
    # Print comprehensive summary
    solver.print_summary(result)
    
    # Save numerical solution
    solver.save_solution(result, "tov_stt_solution")
    
    # Create plots
    solver.plot_solution(result, show_plots=True, save_plots=True, 
                        filename_base="tov_stt_plots")
    
    print(f"\nSolution completed successfully!")
    print(f"Files saved: tov_stt_solution.dat, tov_stt_solution_cgs.dat")
    print(f"Plots saved: tov_stt_plots_profiles.png")


if __name__ == "__main__":
    main()