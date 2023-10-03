#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import pathlib
import numpy as np

import zener

# parse command line arguments using argparse
# input_dir = "./baseline_output"
desc = 'Program to do staggered grid finite difference modeling in 2D fully anisotropic media with attenuation based on relaxation of standard linear solid.'

parser = argparse.ArgumentParser(description=desc, formatter_class = argparse.ArgumentDefaultsHelpFormatter)

requiredNamed = parser.add_argument_group('Required Parameters')
"""
requiredNamed.add_argument("-fpeak", required=True, type=np.float64, \
                    help="Peak frequency of wavelet in Hz")
"""
optionalNamed = parser.add_argument_group('Optional Parameters')

optionalNamed.add_argument("-fpeak", "-fp", type=np.float64, \
                           default=np.float64(40.0), \
                    help="Peak frequency of wavelet in Hz")
optionalNamed.add_argument("-verbose", '-v', type=int, default=0, \
                           help=">0 for verbose output to stderr")
optionalNamed.add_argument("-parmsfile", '-gf', type=str, \
                           default='../data/parms.xlsx', \
                           help="Grid file for input to program")
optionalNamed.add_argument("-wavetype", '-wt', type=int, default=0, \
                           help="0 for Ricker wavelet; 1 for AKB waelet 2;\
                           for MIT style Ricker wavelet")
optionalNamed.add_argument("-wavetime", "-wtm", type=np.float64, \
                           default=np.float64(0.07), \
                           help="time length in seconds of applied wavelet")
optionalNamed.add_argument("-waveletdir", '-wdir', type=str, \
                           default='Wavelet', \
                           help="directory for output ascii wavelet file")
optionalNamed.add_argument("-snapdir", '-sd', type=str, \
                           default='snapshots', \
                           help="directory for output of snapshots")
optionalNamed.add_argument("-tracedir", '-trd', type=str, \
                           default='traces', \
                           help="directory for output of snapshots")
optionalNamed.add_argument("-use_PML", '-pml', type=int, default=1, \
                           help="= 1 to use Perfectly Matched Layer \
                           absorbing boundayr conditions")
optionalNamed.add_argument("-a11file", '-a11f', type=str, \
                           default='a11.bin', \
                           help="files of elastic constants and density \
                           aij = Cij/rho (density normalized stiffness)")
optionalNamed.add_argument("-a13file", '-a13f', type=str, \
                           default='a13.bin', \
                           help="aij = Cij/rho (density normalized stiffness)")
optionalNamed.add_argument("-a15file", '-a15f', type=str, \
                           default='a15.bin', \
                           help="aij = Cij/rho (density normalized stiffness)")
optionalNamed.add_argument("-a33file", '-a33f', type=str, \
                           default='a33.bin', \
                           help="aij = Cij/rho (density normalized stiffness)")
optionalNamed.add_argument("-a35file", '-a35f', type=str, \
                           default='a35.bin', \
                           help="aij = Cij/rho (density normalized stiffness)")
optionalNamed.add_argument("-a55file", '-a55f', type=str, \
                           default='a55.bin', \
                           help="aij = Cij/rho (density normalized stiffness)")
optionalNamed.add_argument("-rhofile", '-rhof', type=str, \
                           default='density.bin', \
                           help="grid of density values")

optionalNamed.add_argument("-tau_sig_dfile", '-tau_sigP', type=str, \
                           default='tau_sig_d.bin', \
                           help="grid of stress dilational (P) \
                           relaxation times")
optionalNamed.add_argument("-tau_eps_dfile", '-tau_epsP', type=str, \
                           default='tau_eps_d.bin', \
                           help="grid of strain dilational (P) \
                           relaxation times")
optionalNamed.add_argument("-tau_sig_sfile", '-tau_sigS', type=str, \
                           default='tau_sig_s.bin', \
                           help="grid of stress shear (S) \
                           relaxation times")
optionalNamed.add_argument("-tau_eps_sfile", '-tau_epsS', type=str, \
                           default='tau_eps_s.bin', \
                           help="grid of strain shear (S) \
                           relaxation times")

args = parser.parse_args()

verbose = args.verbose

arg_items =  args.__dict__.items()

# Assign values to input variables

for item in arg_items:
    locals()[item[0]] = item[1]

# Print Command Line parameters

if verbose > 1:
    print("Input Parameters", file=sys.stderr, flush=True)
    for item in arg_items:
        print('\t{:13s} = {:15s}\ttype = {:s}'.\
              format(item[0], str(item[1]), \
                     str(type(item[1])).split("'")[1]), \
              file=sys.stderr, flush=True)

    print('', file=sys.stderr, flush=True)

grid = zener.Grid_Info(parmsfile)

# Add some input parameter entries to grid object

setattr(grid, 'wavetype', int(wavetype))
setattr(grid, 'fpeak', np.float64(fpeak))
setattr(grid, 'nwav', (round)(wavetime/grid.dt) + 1)

print('', file=sys.stderr, flush=True)

if verbose > 2:
    # list attributes in grid

    grid_attributes = filter(lambda a: not a.startswith('__'), dir(grid))

    for item in grid_attributes:
        val = getattr(grid, item)
        print('\tattribute: {:s} = {:}'.format(item, val), \
              file=sys.stderr, flush=True)

    print('', file=sys.stderr, flush=True)

if verbose >0:

    # Print some parameters to stderr

    print('Wavelet Parameters:', file=sys.stderr, flush=True)
    print('\tPeak Freq. (Hz): {:.1f}\tTime Length: {:.3f}'. \
          format(fpeak, wavetime), file=sys.stderr, flush=True, end='')
    print('\t# of samples: {:d}'.format(grid.nwav), \
          file=sys.stderr, flush=True)
    if wavetype == 2:
        print("\tWavelet type: MIT Style Ricker", file=sys.stderr, flush=True)
    elif wavetype == 1:
        print("\tWavelet type: AKB", file=sys.stderr, flush=True)
    else:
        print("\tWavelet type: Ricker", file=sys.stderr, flush=True)

    print("\nGrid Parameters:", file=sys.stderr, flush=True)
    print("\t# time steps: {:d}\tinc: {:g} (s)".format(grid.nt,grid.dt), \
          file=sys.stderr, flush=True)
    print("\tnx={:d}\t\tnz={:d}".format(grid.nxtot,grid.nztot), \
          file=sys.stderr, flush=True)
    print("\tnxabs={:d}\tnzabs={:d}\tgrid inc.={:f}". \
          format(grid.nxabs,grid.nzabs, grid.dl), file=sys.stderr, flush=True)
    print("\t#snaps: {:d}\ttime[0]: {:g}\ttime[{:d}]: {:g}". \
          format(grid.nts , grid.itout[0] * grid.dt, grid.nts - 1, \
                 (grid.itout[grid.nts - 1]) * grid.dt), \
          file=sys.stderr, flush=True)
    
    print("\t#Sources: {:d}\tSource type: {:s}". \
          format(grid.nsource, grid.source_type), file=sys.stderr, flush=True)
    for i in range(len((grid.isource))):
        print("\t\tsrc# {:d}\t\tx: {:.1f}\tz: {:.1f}". \
              format(i+1, grid.isource[i] * grid.dl, \
                     grid.jsource[i] * grid.dl), file=sys.stderr, flush=True)
        print("\t\t\t\tpol_x: {:.1f}\tpol_z: {:.1f}\tpol_xz: {:.1f}". \
                format(grid.spol_x[i], grid.spol_z[i], grid.spol_xz[i]), \
              file=sys.stderr, flush=True)

    print("\t#Rcvrs: {:d}".format(grid.nrec), file=sys.stderr, flush=True)

    if verbose > 1:
        for i in range(len(grid.i_rec)):
            print("\t\trec# {:4d}\tx: {:.1f}\tz: {:.1f}". \
                  format(i+1, grid.i_rec[i] * grid.dl, \
                         grid.j_rec[i] * grid.dl), file=sys.stderr, flush=True)

# Write ascii wavelet file

zener.write_wavelet_to_file(waveletdir, grid)

# fill Cij matrix components

sx = 0
ex = grid.nxtot

sz = 0
ez = grid.nztot

Cij = zener.Stiffness(sx, ex, sz, ez)

Cij.read_aijfiles(a11file, a13file, a15file, a33file, \
                  a35file, a55file, rhofile, grid, verbose)

# Print column of Cij to stderr

if verbose > 1:
    ic = int((ex+sx)/4)
    for j in range(sz, ez):
        print('\trho[{:d}][{:3d}] = {:.1f}\tVp[{:d}][{:3d}] = {:.1f}'. \
              format(ic, j, Cij.rho[ic][j], ic, j, \
                     np.sqrt( Cij.c33[ic][j] / Cij.rho[ic][j]) ), \
              file=sys.stderr, flush=True, end='')
        print('\tVs[{:d}][{:3d}] = {:.1f}'.format(ic, j, \
                    np.sqrt( Cij.c55[ic][j] / Cij.rho[ic][j] )), \
              file=sys.stderr, flush=True)

Memvar = zener.MemVars(sx, ex, sz, ez)

if verbose > 0:
    print('\nfinished initializing Memory Variables', \
          file=sys.stderr, flush=True)

if verbose > 3:
    ic = int((ex+sx)/4)
    for j in range(sz, ez):
        print('eps1[{:d}][{:3d}] = {:.1f}'. \
              format(ic, j, Memvar.eps1[ic][j]), \
              file=sys.stderr, flush=True, end='') 
        print('  eps2[{:d}][{:3d}] = {:.1f}'. \
              format(ic, j, Memvar.eps2[ic][j]), \
              file=sys.stderr, flush=True, end='') 
        print('  eps2[{:d}][{:3d}] = {:.1f}'. \
              format(ic, j, Memvar.eps3[ic][j]), \
              file=sys.stderr, flush=True) 
# Read tau files

if verbose > 0:
    print('\nReading Tau files', file=sys.stderr, flush=True)

tau = zener.Taus(sx, ex, sz, ez)

tau.fill_Taus (tau_sig_dfile, tau_eps_dfile, \
               tau_sig_sfile, tau_eps_sfile, \
               grid, verbose)

if verbose > 1:
    ic = int((ex+sx)/4)
    for j in range(sz, ez):
        print('  tau[{:d}][{:3d}] sig_d: {:.2g}'. \
              format(ic, j, tau.tau_sig_d[ic][j],), \
              file=sys.stderr, flush=True, end='')
        print('  eps_d: {:.2g}'. \
              format( tau.tau_eps_d[ic][j]), \
              file=sys.stderr, flush=True, end='')
        print('  sig_s: {:.2g}'. \
              format( tau.tau_sig_s[ic][j]), \
              file=sys.stderr, flush=True, end='')
        print('  eps_s: {:.2g}'. \
              format( tau.tau_eps_s[ic][j]), \
              file=sys.stderr, flush=True)

if verbose > 1:
    print('\n\tInitializing C-PML absorbing boundaries', \
          file=sys.stderr, flush=True)

PML = zener.PML_Parms(sx, ex, sz, ez)
PML.initialize_PML_parms(grid, verbose)

"""
sz_bottom = grid.nztot - grid.nzabs - 1
N_bottom = grid.nztot - 1
for j in range(sz_bottom, N_bottom + 1):
    jc = j - sz_bottom
    print('PML.az_half_b[{:d}] = {:g}'.format(j, PML.az_half_b[jc]), file=sys.stderr, flush=True)
"""

# Initialize velocity and stess matrices

vx = np.zeros([grid.nxtot + 4, grid.nztot + 4])
vz = np.zeros([grid.nxtot + 4, grid.nztot + 4])

sxx = np.zeros([grid.nxtot + 4, grid.nztot + 4])
sxz = np.zeros([grid.nxtot + 4, grid.nztot + 4])
szz = np.zeros([grid.nxtot + 4, grid.nztot + 4])

F = zener.Body_Force(grid)

F.get_body_force(sx, ex, sz, ez, Cij)

# Create snapshot, trace directories

if verbose > 0:
    print('\nCreating trace directory: {:s}'.format(tracedir), \
          file=sys.stderr, flush=True)
pathlib.Path(tracedir).mkdir(parents=True, exist_ok=True)
if verbose > 0:
    print('\nCreating snapshot directory: {:s}'.format(snapdir), \
          file=sys.stderr, flush=True)
pathlib.Path(snapdir).mkdir(parents=True, exist_ok=True)

Vxp, Vzp, Pp = zener.open_trace_files(tracedir, verbose)

#------------- Main finite difference time loop -------------------

print('', file=sys.stderr, flush=True)
for it in range(grid.nt):

    if verbose > 0:
        print('Executing time step {:7d}\t time = {:.6f}'. \
              format(it, np.float64(it*grid.dt)), \
              file=sys.stderr, flush=True)
    
    if use_PML:
        vx, vz, \
        PML.Psi_x_sxx_l, \
        PML.Psi_x_sxx_r, \
        PML.Psi_x_sxz_l, \
        PML.Psi_x_sxz_r, \
        PML.Psi_z_sxz_t, \
        PML.Psi_z_sxz_b, \
        PML.Psi_z_szz_t, \
        PML.Psi_z_szz_b = zener.update_velocities_PML(sx, ex, sz, ez, Cij.rho, \
                                                      grid.nxtot, grid.nztot, \
                                                      grid.nxabs, grid.nzabs, \
                                                      grid.dl, grid.dt, \
                                                      sxx, sxz, szz, \
                                                      PML.ax_l, \
                                                      PML.ax_r, \
                                                      PML.ax_half_l, \
                                                      PML.ax_half_r, \
                                                      PML.az_t, \
                                                      PML.az_b, \
                                                      PML.az_half_t, \
                                                      PML.az_half_b, \
                                                      PML.bx_l, \
                                                      PML.bx_r, \
                                                      PML.bx_half_l, \
                                                      PML.bx_half_r, \
                                                      PML.bz_t, \
                                                      PML.bz_b, \
                                                      PML.bz_half_t, \
                                                      PML.bz_half_b, \
                                                      PML.Psi_x_sxx_l, \
                                                      PML.Psi_x_sxx_r, \
                                                      PML.Psi_x_sxz_l, \
                                                      PML.Psi_x_sxz_r, \
                                                      PML.Psi_z_sxz_t, \
                                                      PML.Psi_z_sxz_b, \
                                                      PML.Psi_z_szz_t, \
                                                      PML.Psi_z_szz_b, \
                                                      vx, vz)

    else:
        vx, vz = zener.update_velocities(sx, ex, sz, ez, Cij.rho, \
                                         grid.nxtot, grid.nztot, \
                                         grid.nxabs, grid.nzabs, \
                                         grid.dl, grid.dt, \
                                         sxx, sxz, szz, vx, vz)

    if it < grid.nwav and grid.source_type == 'f':
        vx, vz = zener.apply_body_force(it, sx, ex, sz, ez, \
                                        grid.dl, grid.nsource, \
                                        grid.isource, grid.jsource, \
                                        F.Fx, F.Fz, F.Fxz, \
                                        vx, vz)

    if it < grid.nwav and grid.source_type == 'v':
        vz, vz = zener.apply_velocity_source(it, sx, ex, sz, ez, \
                                             grid.nsource, \
                                             grid.isource, grid.jsource, \
                                             F.Fx, F.Fz, \
                                             vx, vz)
    if use_PML:
        sxx, sxz, szz, \
        Memvar.eps1, \
        Memvar.eps2, \
        Memvar.eps3, \
        PML.Psi_x_vx_l, \
        PML.Psi_x_vx_r, \
        PML.Psi_x_vz_l, \
        PML.Psi_x_vz_r, \
        PML.Psi_z_vx_t, \
        PML.Psi_z_vx_b, \
        PML.Psi_z_vz_t, \
        PML.Psi_z_vz_b = zener.update_stresses_PML(sx, ex, sz, ez, \
                                                   Cij.c11, Cij.c13, Cij.c15, \
                                                   Cij.c33, Cij.c35, Cij.c55, \
                                                   Cij.rho, \
                                                   tau.tau_sig_d, \
                                                   tau.tau_eps_d, \
                                                   tau.tau_sig_s, \
                                                   tau.tau_eps_s, \
                                                   grid.nxtot, grid.nztot, \
                                                   grid.nxabs, grid.nzabs, \
                                                   grid.dl, grid.dt,  \
                                                   vx, vz, \
                                                   PML.ax_l, \
                                                   PML.ax_r, \
                                                   PML.ax_half_l, \
                                                   PML.ax_half_r, \
                                                   PML.az_t, \
                                                   PML.az_b, \
                                                   PML.az_half_t, \
                                                   PML.az_half_b, \
                                                   PML.bx_l, \
                                                   PML.bx_r, \
                                                   PML.bx_half_l, \
                                                   PML.bx_half_r, \
                                                   PML.bz_t, \
                                                   PML.bz_b, \
                                                   PML.bz_half_t, \
                                                   PML.bz_half_b, \
                                                   PML.Psi_x_vx_l, \
                                                   PML.Psi_x_vx_r, \
                                                   PML.Psi_x_vz_l, \
                                                   PML.Psi_x_vz_r, \
                                                   PML.Psi_z_vx_t, \
                                                   PML.Psi_z_vx_b, \
                                                   PML.Psi_z_vz_t, \
                                                   PML.Psi_z_vz_b, \
                                                   Memvar.eps1, \
                                                   Memvar.eps2, \
                                                   Memvar.eps3, \
                                                   sxx, sxz, szz)
    else:
        sxx, sxz, szz = zener.update_stresses(sx, ex, sz, ez, \
                                              Cij.c11, Cij.c13, Cij.c15, \
                                              Cij.c33, Cij.c35, Cij.c55, \
                                              Cij.rho, \
                                              grid.nxtot, grid.nztot, \
                                              grid.nxabs, grid.nzabs, \
                                              grid.dl, grid.dt,  \
                                              vx, vz, sxx, sxz, szz)

    if grid.source_type == 't' and it < grid.nwav:
        sxx, sxz, szz = zener.apply_stress_source(it, sx, ex, sz, ez, \
                                                  grid.nsource, \
                                                  grid.isource, grid.jsource, \
                                                  F.Fx, F.Fz, F.Fxz, \
                                                  sxx, sxz, szz)
    for i in range(grid.nts):
        if it == grid.itout[i]:
            zener.write_snapshot(it, sx, ex, sz, ez, grid, verbose, \
                                 snapdir, vx, vz, sxx, sxz, szz)

    zener.write_traces(sx, ex, sz, ez, grid, vz, vz, sxx, sxz, szz, \
                       Vxp, Vzp, Pp)

       

    

#------------- End of finite difference time loop -------------------


zener.close_trace_files(Vxp, Vzp, Pp)
