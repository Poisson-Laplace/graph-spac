# author: ferat
# date: 2026
# vol4 — Geometry Transforms

import numpy as np

def poincare_to_physical(u, v, x_max, y_max):
    """
    Maps points from the Poincare Unit Disk (u^2 + v^2 <= 1)
    to physical Euclidean space [0, x_max] x [0, y_max].
    
    The mapping expands the center and compresses the edges in the physical space,
    or rather, if points are uniform in the disk, they become dense in the center 
    and sparse at the edges of the physical space (similar to a Kennett Spiral).
    """
    u = np.clip(u, -0.999, 0.999)
    v = np.clip(v, -0.999, 0.999)
    
    r_hyper = np.sqrt(u**2 + v**2)
    # To prevent division by zero or log(0)
    r_hyper = np.clip(r_hyper, 1e-6, 0.999)
    
    # Non-linear mapping: r_phys grows logarithmically / hyperbolically
    # We want r_hyper = 0 -> r_phys = 0
    # We want r_hyper = 1 -> r_phys = R_max
    
    # Distance in Poincare disk from origin: d = 2 * arctanh(r)
    d_hyper = 2 * np.arctanh(r_hyper)
    
    # Map maximum possible distance (at r=0.999) to physical R_max
    max_d = 2 * np.arctanh(0.999)
    
    cx, cy = x_max / 2.0, y_max / 2.0
    R_max = min(cx, cy)
    
    r_phys = R_max * (d_hyper / max_d)
    
    theta = np.arctan2(v, u)
    
    x_phys = cx + r_phys * np.cos(theta)
    y_phys = cy + r_phys * np.sin(theta)
    
    return x_phys, y_phys

def apply_transforms(coords_hyperbolic, x_max, y_max, use_poincare=False):
    """
    Applies active geometric transforms (Poincare -> Physical).
    """
    if not use_poincare:
        return coords_hyperbolic
        
    coords_phys = np.empty_like(coords_hyperbolic)
    for i in range(len(coords_hyperbolic)):
        u, v = coords_hyperbolic[i]
        px, py = poincare_to_physical(u, v, x_max, y_max)
        coords_phys[i, 0] = px
        coords_phys[i, 1] = py
        
    return coords_phys
