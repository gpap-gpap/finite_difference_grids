
"""
Guided by:

https://www.w3schools.com/python/python_classes.asp
and
https://pynative.com/python-class-variables/
"""

import pandas as pd
import numpy as np
import sys
import os
import pathlib
import io
import math
from numba import jit

# class Grid_Info_Better:
#     # No class variables
#     def __init__(self, parmsfile: str):
#         self.parmsfile = parmsfile
#         self._parms_dict = None
    
#     @property
#     def parms_dict(self):
#         if self._parms_dict is None:
#             self._parms_dict = 

#     def read_grid_file(self, parmsfile: str):
#         xlsx_path = pd.ExcelFile(parmsfile)
#         df = pd.read_excel(xlsx_path)
#         parm_dict = dict(zip(df.Parameter, df.Value))
#         return(parm_dict)

#         #----------------------------------------------------

#         def rcvr_arrays(self, \
#                         start_rcvr: float, end_rcvr: float, rcvr_z: float, \
#                         del_rcvr, vsp_x: float, vsp_sz: float, vsp_ez: float, \
#                         del_vsp: float, nxtot: int, nztot: int, dl: float):
            
#             # Create empty i_rec, j_rec np arrays

#             i_rec = np.zeros([0,], dtype=int)
#             j_rec = np.zeros([0,], dtype=int)
            
#             # Surface receivers

#             srf_grid_z = []
#             srf_grid_start = []
#             srf_grid_end = []
#             num_surf = []
#             num_rcvr = 0
#             for i in range(len(rcvr_z)):
#                 srf_grid_z.append(nztot - round(rcvr_z[i]/dl))
#                 srf_grid_start.append(1 + round(start_rcvr[i]/dl))
#                 srf_grid_end.append(1 + round(end_rcvr[i]/dl))
#                 num_surf.append(1 + \
#                     round( (srf_grid_end[i] - srf_grid_start[i]) / \
#                            (del_rcvr[i]/dl) ))
#                 num_rcvr = num_rcvr + num_surf[i]

#                 i_surf = np.arange(srf_grid_start[i], 1+srf_grid_end[i], \
#                                    round(del_rcvr[i]/dl), dtype=int)
                
#                 j_surf = np.ones([num_surf[i],], dtype=int)*srf_grid_z

#                 i_rec = np.append(i_rec, i_surf)
#                 j_rec = np.append(j_rec, j_surf)

#             # vsp receivers

#             vsp_grid_sz = []
#             vsp_grid_ez = []
#             num_vsp = []
#             num_vsp_tot = 0
#             for i in range(len(vsp_x)):
#                 vsp_grid_sz.append(nztot - round(vsp_sz[i]/dl))
#                 vsp_grid_ez.append(nztot - round(vsp_ez[i]/dl))
#                 num_vsp.append(1 + round( (vsp_grid_sz[i] - vsp_grid_ez[i]) / \
#                                           (del_vsp[i]/dl) ) )
#                 num_vsp_tot = num_vsp_tot + num_vsp[i]
#                 num_rcvr = num_rcvr + num_vsp[i]

#                 i_vsp = np.ones([num_vsp[i],], dtype=int)* \
#                     (1 + round((vsp_x[i]/dl)))

#                 # Decreasing np.arange()
#                 j_vsp = np.arange(vsp_grid_sz[i], vsp_grid_ez[i]-1, \
#                                   round(-del_vsp[i]/dl), dtype=int)

#                 i_rec = np.append(i_rec, i_vsp)
#                 j_rec = np.append(j_rec, j_vsp)

#             # Make z correction for grid 0 at bottom
#             # adjust x to start at 0
#             for i in range(len(j_rec)):
#                 i_rec[i] = i_rec[i] - 1
#                 j_rec[i] = nztot - j_rec[i]
           
#             return i_rec, j_rec

#         #----------------------------------------------------

#         def apply_gaussian_weigths(self, nsource: int, \
#                                    x_pol: np.ndarray, \
#                                    z_pol: np.ndarray, \
#                                    isrc: np.ndarray, \
#                                    jsrc: np.ndarray):

#             nx = 3
#             nz = 3

#             sx = isrc[0]
#             sz = jsrc[0]

#             amp0_x = x_pol[0]
#             amp0_z = z_pol[0]

#             x_pol = amp0_x*np.ones([1,], dtype=float)
#             z_pol = amp0_z*np.ones([1,], dtype=float)

#             for ix in range(0, nx):
#                 for iz in range(0, nz):
#                     if not (ix == 1 and iz == 1):
#                         i_new = np.array([sx+ix-1])
#                         j_new = np.array([sz+iz-1])

#                         x = float(ix-1)
#                         z = float(iz-1)
#                         r = np.sqrt( (x)*(x) + (z)*(z) )
#                         x_pol_new = amp0_x*np.exp(-r*r/2.0)
#                         z_pol_new = amp0_z*np.exp(-r*r/2.0)
                        
#                         isrc = np.append(isrc, i_new)
#                         jsrc = np.append(jsrc, j_new)

#                         x_pol = np.append(x_pol, x_pol_new)
#                         z_pol = np.append(z_pol, z_pol_new)

#             return x_pol, z_pol, isrc, jsrc

#         #----------------------------------------------------

#         def source_array (self, \
#                           nsource: int, x_source: float, z_source: float , \
#                           stype: int, nxtot: int, nztot: int, dl: float):

#             # Create empty isource, jsource spol np arrays

#             isource = np.zeros([0,], dtype=int)
#             jsource = np.zeros([0,], dtype=int)
            
#             if stype == 0:
#                 source_type = 't'
#                 spol_x = np.ones([nsource,], dtype=float)
#                 spol_z = np.ones([nsource,], dtype=float)
#                 spol_xz = np.zeros([nsource,], dtype=float)
#             elif stype == 1:
#                 source_type = 'v'
#                 spol_x = np.ones([nsource,], dtype=float)
#                 spol_z = np.zeros([nsource,], dtype=float)
#                 spol_xz = np.zeros([nsource,], dtype=float)
#             elif stype == 2:
#                 source_type = 'v'
#                 spol_x = np.zeros([nsource,], dtype=float)
#                 spol_z = np.ones([nsource,], dtype=float)
#                 spol_xz = np.zeros([nsource,], dtype=float)
#             else:
#                 print('\n\tERROR: stype {:d} NOT IMPLEMENTED\n'.format(stype), \
#                       file=sys.stderr, flush=True)
#                 quit()

#             src_grid_z = []
#             src_grid_x = []
#             for i in range(len(z_source)):
#                 src_grid_z.append(nztot - (z_source[i]/dl))
#                 src_grid_x.append(nxtot - (x_source[i]/dl))
 
#                 isource = np.append(isource, np.array([src_grid_x], dtype=int))
#                 jsource = np.append(jsource, np.array([src_grid_z], dtype=int))

#             if nsource % 9 == 0:
#                 spol_x, spol_z, isource, jsource = \
#                     apply_gaussian_weigths(self, nsource, \
#                                            spol_x, spol_z, isource, jsource)

#             # Make z correction for grid 0 at bottom
#             # adjust x to start at 0
#             for i in range(len(jsource)):
#                 isource[i] = isource[i] - 1
#                 jsource[i] = nztot - jsource[i]

#             return source_type, isource, jsource, spol_x, spol_z, spol_xz

        

#         #----------------------------------------------------
        
#         def convert_to_float_list(self, parm):
#             if str(parm).find(' ') > 0:
#                 parm = list((parm.split(" ")))
#             else:
#                 parm = [parm]
#             parm = [np.float64(x) for x in parm]
#             return(parm)

#         #----------------------------------------------------
            
#         # Set Instance variables

#         parm_dict = read_grid_file(self, self.parmsfile)

#         # Convert parm.dict to class instance attributes

#         for key in parm_dict:
#             setattr(self, key, parm_dict[key])

#         # Loop over array parameters converting to float lists
        
#         self.start_rcvr = convert_to_float_list(self, self.start_rcvr)
#         self.end_rcvr = convert_to_float_list(self, self.end_rcvr)
#         self.rcvr_z = convert_to_float_list(self, self.rcvr_z)
#         self.del_rcvr = convert_to_float_list(self, self.del_rcvr)

#         self.vsp_x = convert_to_float_list(self, self.vsp_x)
#         self.vsp_sz = convert_to_float_list(self, self.vsp_sz)
#         self.vsp_ez = convert_to_float_list(self, self.vsp_ez)
#         self.del_vsp = convert_to_float_list(self, self.del_vsp)
        
#         self.x_source = convert_to_float_list(self, self.x_source)
#         self.z_source = convert_to_float_list(self, self.z_source)
#         """
#         if str(self.x_source).find(' ') > 0:
#             self.x_source = list((self.z_source.split(" ")))
#             self.x_source = [np.float64(x) for x in self.x_source]
        
#         if str(self.z_source).find(' ') > 0:
#             self.z_source = list((self.z_source.split(" ")))
#             self.z_source = [np.float64(x) for x in self.z_source]
#         """
#         # Generate the itout list of time step indices to output snapshot

#         self.itout = np.arange(self.fsnap, \
#                                self.fsnap + self.nsnap*self.dsnap, \
#                                self.dsnap, dtype=int)

#         self.nts = self.nsnap
          
#         # Generate i_rec, j_rec arrays

#         self.i_rec, self.j_rec = rcvr_arrays(self, \
#                                              self.start_rcvr, \
#                                              self.end_rcvr, self.rcvr_z, \
#                                              self.del_rcvr, self.vsp_x, \
#                                              self.vsp_sz, self.vsp_ez, \
#                                              self.del_vsp, \
#                                              self.nxtot, self.nztot, self.dl)

#         self.nrec = len(self.i_rec)

#         self.source_type, \
#             self.isource, \
#             self.jsource, \
#             self.spol_x, \
#             self.spol_z, \
#             self.spol_xz = source_array(self, \
#                                         self.nsource, self.x_source, \
#                                         self.z_source, self.stype, \
#                                         self.nxtot, self.nztot, self.dl)



class Grid_Info:
    # No class variables
    def __init__(self, parmsfile: str):
        self.parmsfile = parmsfile

        def read_grid_file(self, parmsfile: str):
            xlsx_path = pd.ExcelFile(parmsfile)
            df = pd.read_excel(xlsx_path)
            parm_dict = dict(zip(df.Parameter, df.Value))
            return(parm_dict)

        #----------------------------------------------------

        def rcvr_arrays(self, \
                        start_rcvr: float, end_rcvr: float, rcvr_z: float, \
                        del_rcvr, vsp_x: float, vsp_sz: float, vsp_ez: float, \
                        del_vsp: float, nxtot: int, nztot: int, dl: float):
            
            # Create empty i_rec, j_rec np arrays

            i_rec = np.zeros([0,], dtype=int)
            j_rec = np.zeros([0,], dtype=int)
            
            # Surface receivers

            srf_grid_z = []
            srf_grid_start = []
            srf_grid_end = []
            num_surf = []
            num_rcvr = 0
            for i in range(len(rcvr_z)):
                srf_grid_z.append(nztot - round(rcvr_z[i]/dl))
                srf_grid_start.append(1 + round(start_rcvr[i]/dl))
                srf_grid_end.append(1 + round(end_rcvr[i]/dl))
                num_surf.append(1 + \
                    round( (srf_grid_end[i] - srf_grid_start[i]) / \
                           (del_rcvr[i]/dl) ))
                num_rcvr = num_rcvr + num_surf[i]

                i_surf = np.arange(srf_grid_start[i], 1+srf_grid_end[i], \
                                   round(del_rcvr[i]/dl), dtype=int)
                
                j_surf = np.ones([num_surf[i],], dtype=int)*srf_grid_z

                i_rec = np.append(i_rec, i_surf)
                j_rec = np.append(j_rec, j_surf)

            # vsp receivers

            vsp_grid_sz = []
            vsp_grid_ez = []
            num_vsp = []
            num_vsp_tot = 0
            for i in range(len(vsp_x)):
                vsp_grid_sz.append(nztot - round(vsp_sz[i]/dl))
                vsp_grid_ez.append(nztot - round(vsp_ez[i]/dl))
                num_vsp.append(1 + round( (vsp_grid_sz[i] - vsp_grid_ez[i]) / \
                                          (del_vsp[i]/dl) ) )
                num_vsp_tot = num_vsp_tot + num_vsp[i]
                num_rcvr = num_rcvr + num_vsp[i]

                i_vsp = np.ones([num_vsp[i],], dtype=int)* \
                    (1 + round((vsp_x[i]/dl)))

                # Decreasing np.arange()
                j_vsp = np.arange(vsp_grid_sz[i], vsp_grid_ez[i]-1, \
                                  round(-del_vsp[i]/dl), dtype=int)

                i_rec = np.append(i_rec, i_vsp)
                j_rec = np.append(j_rec, j_vsp)

            # Make z correction for grid 0 at bottom
            # adjust x to start at 0
            for i in range(len(j_rec)):
                i_rec[i] = i_rec[i] - 1
                j_rec[i] = nztot - j_rec[i]
           
            return i_rec, j_rec

        #----------------------------------------------------

        def apply_gaussian_weigths(self, nsource: int, \
                                   x_pol: np.ndarray, \
                                   z_pol: np.ndarray, \
                                   isrc: np.ndarray, \
                                   jsrc: np.ndarray):

            nx = 3
            nz = 3

            sx = isrc[0]
            sz = jsrc[0]

            amp0_x = x_pol[0]
            amp0_z = z_pol[0]

            x_pol = amp0_x*np.ones([1,], dtype=float)
            z_pol = amp0_z*np.ones([1,], dtype=float)

            for ix in range(0, nx):
                for iz in range(0, nz):
                    if not (ix == 1 and iz == 1):
                        i_new = np.array([sx+ix-1])
                        j_new = np.array([sz+iz-1])

                        x = float(ix-1)
                        z = float(iz-1)
                        r = np.sqrt( (x)*(x) + (z)*(z) )
                        x_pol_new = amp0_x*np.exp(-r*r/2.0)
                        z_pol_new = amp0_z*np.exp(-r*r/2.0)
                        
                        isrc = np.append(isrc, i_new)
                        jsrc = np.append(jsrc, j_new)

                        x_pol = np.append(x_pol, x_pol_new)
                        z_pol = np.append(z_pol, z_pol_new)

            return x_pol, z_pol, isrc, jsrc

        #----------------------------------------------------

        def source_array (self, \
                          nsource: int, x_source: float, z_source: float , \
                          stype: int, nxtot: int, nztot: int, dl: float):

            # Create empty isource, jsource spol np arrays

            isource = np.zeros([0,], dtype=int)
            jsource = np.zeros([0,], dtype=int)
            
            if stype == 0:
                source_type = 't'
                spol_x = np.ones([nsource,], dtype=float)
                spol_z = np.ones([nsource,], dtype=float)
                spol_xz = np.zeros([nsource,], dtype=float)
            elif stype == 1:
                source_type = 'v'
                spol_x = np.ones([nsource,], dtype=float)
                spol_z = np.zeros([nsource,], dtype=float)
                spol_xz = np.zeros([nsource,], dtype=float)
            elif stype == 2:
                source_type = 'v'
                spol_x = np.zeros([nsource,], dtype=float)
                spol_z = np.ones([nsource,], dtype=float)
                spol_xz = np.zeros([nsource,], dtype=float)
            else:
                print('\n\tERROR: stype {:d} NOT IMPLEMENTED\n'.format(stype), \
                      file=sys.stderr, flush=True)
                quit()

            src_grid_z = []
            src_grid_x = []
            for i in range(len(z_source)):
                src_grid_z.append(nztot - (z_source[i]/dl))
                src_grid_x.append(nxtot - (x_source[i]/dl))
 
                isource = np.append(isource, np.array([src_grid_x], dtype=int))
                jsource = np.append(jsource, np.array([src_grid_z], dtype=int))

            if nsource % 9 == 0:
                spol_x, spol_z, isource, jsource = \
                    apply_gaussian_weigths(self, nsource, \
                                           spol_x, spol_z, isource, jsource)

            # Make z correction for grid 0 at bottom
            # adjust x to start at 0
            for i in range(len(jsource)):
                isource[i] = isource[i] - 1
                jsource[i] = nztot - jsource[i]

            return source_type, isource, jsource, spol_x, spol_z, spol_xz

        

        #----------------------------------------------------
        
        def convert_to_float_list(self, parm):
            if str(parm).find(' ') > 0:
                parm = list((parm.split(" ")))
            else:
                parm = [parm]
            parm = [np.float64(x) for x in parm]
            return(parm)

        #----------------------------------------------------
            
        # Set Instance variables

        parm_dict = read_grid_file(self, self.parmsfile)

        # Convert parm.dict to class instance attributes

        for key in parm_dict:
            setattr(self, key, parm_dict[key])

        # Loop over array parameters converting to float lists
        
        self.start_rcvr = convert_to_float_list(self, self.start_rcvr)
        self.end_rcvr = convert_to_float_list(self, self.end_rcvr)
        self.rcvr_z = convert_to_float_list(self, self.rcvr_z)
        self.del_rcvr = convert_to_float_list(self, self.del_rcvr)

        self.vsp_x = convert_to_float_list(self, self.vsp_x)
        self.vsp_sz = convert_to_float_list(self, self.vsp_sz)
        self.vsp_ez = convert_to_float_list(self, self.vsp_ez)
        self.del_vsp = convert_to_float_list(self, self.del_vsp)
        
        self.x_source = convert_to_float_list(self, self.x_source)
        self.z_source = convert_to_float_list(self, self.z_source)
        """
        if str(self.x_source).find(' ') > 0:
            self.x_source = list((self.z_source.split(" ")))
            self.x_source = [np.float64(x) for x in self.x_source]
        
        if str(self.z_source).find(' ') > 0:
            self.z_source = list((self.z_source.split(" ")))
            self.z_source = [np.float64(x) for x in self.z_source]
        """
        # Generate the itout list of time step indices to output snapshot

        self.itout = np.arange(self.fsnap, \
                               self.fsnap + self.nsnap*self.dsnap, \
                               self.dsnap, dtype=int)

        self.nts = self.nsnap
          
        # Generate i_rec, j_rec arrays

        self.i_rec, self.j_rec = rcvr_arrays(self, \
                                             self.start_rcvr, \
                                             self.end_rcvr, self.rcvr_z, \
                                             self.del_rcvr, self.vsp_x, \
                                             self.vsp_sz, self.vsp_ez, \
                                             self.del_vsp, \
                                             self.nxtot, self.nztot, self.dl)

        self.nrec = len(self.i_rec)

        self.source_type, \
            self.isource, \
            self.jsource, \
            self.spol_x, \
            self.spol_z, \
            self.spol_xz = source_array(self, \
                                        self.nsource, self.x_source, \
                                        self.z_source, self.stype, \
                                        self.nxtot, self.nztot, self.dl)

#--------------- End of Class GridInfo --------------------

def ricker1_wavelet(nt: int, dt: np.float64, fpeak: np.float64):

    wavelet = np.zeros([0,], dtype=np.float64)
    t0 = 1.0/fpeak
    for it in range(nt):
        t1=it*dt
        arg1 = -np.pi*np.pi*fpeak*fpeak*(t1-t0)*(t1-t0)
        arg2 = 1.0-2.*np.pi*np.pi*fpeak*fpeak*(t1-t0)*(t1-t0)
        amp = np.array(np.exp(arg1)) * arg2
        wavelet = np.append(wavelet, amp)
    return wavelet

#--------------- End of function ricker1_wavelet --------------------

def akb_wavelet (nt: int, dt: np.float64, fpeak: np.float64):

    """
    max amplitude without normalization is exp(-1/2)/(2*fpeak*PI)
    = 0.096532353/fpeak
    """
    wavelet = np.zeros([0,], dtype=np.float64)
    t0 = 1.0/fpeak
    for it in range(nt):
        t1 = it*dt
        term1 = -(fpeak/0.096532353)*(t1-t0)
        term2 = -2.0*fpeak*fpeak*np.pi*np.pi*(t1-t0)*(t1-t0)
        amp = np.array(term1*np.exp(term2))
        wavelet = np.append(wavelet, amp)
    return wavelet

#--------------- End of function akb_wavelet --------------------

def rickerMIT_wavelet (nt: int, dt: np.float64, fpeak: np.float64):
    wavelet = np.zeros([0,], dtype=np.float64)
    t0=1.5/fpeak
    a = fpeak * np.sqrt(2.0)*np.pi
    for it in range(nt):
        t1=(it*dt) - t0
        term1 = a*a*(1 - a*a*t1*t1)
        term2 = -a*a*t1*t1/2.0
        amp = np.array(term1*np.exp(term2))
        wavelet = np.append(wavelet, amp)
    return wavelet

#--------------- End of function rickerMIT_wavelet --------------------

def write_wavelet_to_file(dir: str, grid: Grid_Info):
    # Create dir
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)

    wavetype = grid.wavetype

    file = dir + '/Wavelet-{:d}.txt'.format(wavetype)

    f = open(file, 'w')

    if wavetype == 2:
        f.write('\tWavelet type: MIT Style Ricker fpeak = {:.1f} Hz\n'.\
                format(grid.fpeak))
        wavelet = rickerMIT_wavelet(grid.nwav, grid.dt, grid.fpeak)
    elif wavetype == 1:
        f.write('\tWavelet type: AKB  fpeak = {:.1f} Hz\n'.\
                format(grid.fpeak))
        wavelet = akb_wavelet(grid.nwav, grid.dt, grid.fpeak)
    else:
         f.write('\tWavelet type: Ricker  fpeak = {:.1f} Hz\n'.\
                format(grid.fpeak))
         wavelet = ricker1_wavelet(grid.nwav, grid.dt, grid.fpeak)
    for i in range(grid.nwav):
        f.write('{:f} {:f}\n'.format(float(i*grid.dt), wavelet[i]))
    f.close()

#--------------- End of function write_wavelet_to_file --------------------

class Stiffness:
    def __init__(self, sx: int, ex: int, sz: int, ez: int):
        
        self.c11 = np.zeros([ex-sx, ez-sz], dtype=np.float64)
        self.c13 = np.zeros([ex-sx, ez-sz], dtype=np.float64)
        self.c15 = np.zeros([ex-sx, ez-sz], dtype=np.float64)
        self.c33 = np.zeros([ex-sx, ez-sz], dtype=np.float64)
        self.c35 = np.zeros([ex-sx, ez-sz], dtype=np.float64)
        self.c55 = np.zeros([ex-sx, ez-sz], dtype=np.float64)
        self.rho = np.zeros([ex-sx, ez-sz], dtype=np.float64)

        self.sx = sx
        self.ex = ex
        self.sz = sz
        self.ez = ez

    def read_aijfiles (self, \
                       a11file: str, a13file: str, \
                       a15file: str, a33file: str, \
                       a35file: str, a55file: str, \
                       rhofile: str,
                       grid: Grid_Info, verbose: int):

        sx = self.sx
        ex = self.ex
        sz = self.sz
        ez = self.ez
        
        # Open aij files for reading

        try:
            rhop = open(rhofile, 'rb')
        except FileNotFoundError:
            print('\n\tERROR: Can not open file {:s} - Quitting\n'. \
                  format(os.path.basename(rhofile)), \
                  file=sys.stderr, flush=True)
            quit()

        try:
            a11p = open(a11file, 'rb')
        except FileNotFoundError:
            print('\n\tERROR: Can not open file {:s} - Quitting\n'. \
                  format(os.path.basename(a11file)), \
                  file=sys.stderr, flush=True)
            quit()

        try:
            a13p = open(a13file, 'rb')
        except FileNotFoundError:
            print('\n\tERROR: Can not open file {:s} - Quitting\n'. \
                  format(os.path.basename(a13file)), \
                  file=sys.stderr, flush=True)
            quit()

        try:
            a15p = open(a15file, 'rb')
        except FileNotFoundError:
            print('\n\tERROR: Can not open file {:s} - Quitting\n'. \
                  format(os.path.basename(a15file)), \
                  file=sys.stderr, flush=True)
            quit()

        try:
            a33p = open(a33file, 'rb')
        except FileNotFoundError:
            print('\n\tERROR: Can not open file {:s} - Quitting\n'. \
                  format(os.path.basename(a33file)), \
                  file=sys.stderr, flush=True)
            quit()

        try:
            a35p = open(a35file, 'rb')
        except FileNotFoundError:
            print('\n\tERROR: Can not open file {:s} - Quitting\n'. \
                  format(os.path.basename(a35file)), \
                  file=sys.stderr, flush=True)
            quit()

        try:
            a55p = open(a55file, 'rb')
        except FileNotFoundError:
            print('\n\tERROR: Can not open file {:s} - Quitting\n'. \
                  format(os.path.basename(a55file)), \
                  file=sys.stderr, flush=True)
            quit()

        if verbose > 0:
            print('\n\tSetting up Cij\'s', file=sys.stderr, flush=True)

        file_l = ez - sz

        for i in range(sx, ex):
            file_s = (grid.nztot * i + sz)*4
        
            rhop.seek(file_s, os.SEEK_SET)
            atemp = np.fromfile(rhop, dtype=np.float32, count=file_l)
            
            for j in range(sz, ez):
                self.rho[i][j] = np.float64(atemp[j])
            
            a11p.seek(file_s, os.SEEK_SET)
            atemp = np.fromfile(a11p, dtype=np.float32, count=file_l)
        
            for j in range(sz, ez):
                self.c11[i][j] = np.float64(atemp[j]) * self.rho[i][j]
            
            a13p.seek(file_s, os.SEEK_SET)
            atemp = np.fromfile(a13p, dtype=np.float32, count=file_l)
        
            for j in range(sz, ez):
                self.c13[i][j] = np.float64(atemp[j]) * self.rho[i][j]
           
            a15p.seek(file_s, os.SEEK_SET)
            atemp = np.fromfile(a15p, dtype=np.float32, count=file_l)
        
            for j in range(sz, ez):
                self.c15[i][j] = np.float64(atemp[j]) * self.rho[i][j]
           
            a33p.seek(file_s, os.SEEK_SET)
            atemp = np.fromfile(a33p, dtype=np.float32, count=file_l)
        
            for j in range(sz, ez):
                self.c33[i][j] = np.float64(atemp[j]) * self.rho[i][j]
           
            a35p.seek(file_s, os.SEEK_SET)
            atemp = np.fromfile(a35p, dtype=np.float32, count=file_l)
        
            for j in range(sz, ez):
                self.c35[i][j] = np.float64(atemp[j]) * self.rho[i][j]
           
            a55p.seek(file_s, os.SEEK_SET)
            atemp = np.fromfile(a55p, dtype=np.float32, count=file_l)
        
            for j in range(sz, ez):
                self.c55[i][j] = np.float64(atemp[j]) * self.rho[i][j]

        rhop.close()
        a11p.close()
        a13p.close()
        a15p.close()
        a33p.close()
        a35p.close()
        a55p.close()

#-------------------- End of function fill_Stiffness --------------------

#--------------- End of Class Stiffness ---------------------------------

class MemVars:
    def __init__(self, sx: int, ex: int, sz: int, ez: int):
        self.eps1 = np.zeros([ex-sx, ez-sz], dtype=np.float64)
        self.eps2 = np.zeros([ex-sx, ez-sz], dtype=np.float64)
        self.eps3 = np.zeros([ex-sx, ez-sz], dtype=np.float64)

#--------------- End of class MemVars --------------------
   

class Taus:
    def __init__(self, sx: int, ex: int, sz: int, ez: int):
        # stress dilational relaxation times
        self.tau_sig_d = np.zeros([ex-sx, ez-sz], dtype=np.float64)
        # strain dilational relaxation times
        self.tau_eps_d = np.zeros([ex-sx, ez-sz], dtype=np.float64)
        # stress shear relaxation times
        self.tau_sig_s = np.zeros([ex-sx, ez-sz], dtype=np.float64)
        # strain shear relaxation times
        self.tau_eps_s = np.zeros([ex-sx, ez-sz], dtype=np.float64)

        self.sx = sx
        self.ex = ex
        self.sz = sz
        self.ez = ez

    def fill_Taus (self, \
               tau_sig_dfile: str, tau_eps_dfile: str, \
               tau_sig_sfile: str, tau_eps_sfile: str, \
               grid: Grid_Info, verbose: int):
    
        sx = self.sx
        ex = self.ex
        sz = self.sz
        ez = self.ez
        
        # Open tau files for reading

        try:
            tau_sig_dp = open(tau_sig_dfile, 'rb')
        except FileNotFoundError:
            print('\n\tERROR: Can not open file {:s} - Quitting\n'. \
                  format(os.path.basename(tau_sig_dfile)), \
                  file=sys.stderr, flush=True)
            quit()

        try:
            tau_eps_dp = open(tau_eps_dfile, 'rb')
        except FileNotFoundError:
            print('\n\tERROR: Can not open file {:s} - Quitting\n'. \
                  format(os.path.basename(tau_eps_dfile)), \
                  file=sys.stderr, flush=True)
            quit()

        try:
            tau_sig_sp = open(tau_sig_sfile, 'rb')
        except FileNotFoundError:
            print('\n\tERROR: Can not open file {:s} - Quitting\n'. \
                  format(os.path.basename(tau_sig_sfile)), \
                  file=sys.stderr, flush=True)
            quit()

        try:
            tau_eps_sp = open(tau_eps_sfile, 'rb')
        except FileNotFoundError:
            print('\n\tERROR: Can not open file {:s} - Quitting\n'. \
                  format(os.path.basename(tau_eps_sfile)), \
                  file=sys.stderr, flush=True)
            quit()

        if verbose > 0:
            print('\nSetting up Tau\'s', \
                  file=sys.stderr, flush=True)

        file_l = ez - sz

        for i in range(sx, ex):
            file_s = (grid.nztot * i + sz)*4
            
            tau_sig_dp.seek(file_s, os.SEEK_SET)
            atemp = np.fromfile(tau_sig_dp, dtype=np.float32, count=file_l)
        
            for j in range(sz, ez):
                self.tau_sig_d[i][j] = np.float64(atemp[j])
           
            tau_eps_dp.seek(file_s, os.SEEK_SET)
            atemp = np.fromfile(tau_eps_dp, dtype=np.float32, count=file_l)
        
            for j in range(sz, ez):
                self.tau_eps_d[i][j] = np.float64(atemp[j])
            
            tau_sig_dp.seek(file_s, os.SEEK_SET)
            atemp = np.fromfile(tau_sig_dp, dtype=np.float32, count=file_l)
        
            for j in range(sz, ez):
                self.tau_sig_s[i][j] = np.float64(atemp[j])
           
            tau_eps_sp.seek(file_s, os.SEEK_SET)
            atemp = np.fromfile(tau_eps_sp, dtype=np.float32, count=file_l)
        
            for j in range(sz, ez):
                self.tau_eps_s[i][j] = np.float64(atemp[j])

        if verbose > 0:
            print('\nTau Matrix filled', file=sys.stderr, flush=True)

        tau_sig_dp.close()
        tau_eps_dp.close()
        tau_sig_sp.close()
        tau_eps_sp.close()


        #--------------- End of function fill_Taus --------------------

#--------------- End of class Taus --------------------


class PML_Parms:
    def __init__(self,  sx: int, ex: int, sz: int, ez: int):
        
        self.ax_l = np.zeros([0,], dtype=np.float64)
        self.ax_r = np.zeros([0,], dtype=np.float64)
        self.az_t = np.zeros([0,], dtype=np.float64)
        self.az_b = np.zeros([0,], dtype=np.float64)
        self.az_half_t = np.zeros([0,], dtype=np.float64)
        self.az_half_b = np.zeros([0,], dtype=np.float64)
        self.bx_l = np.zeros([0,], dtype=np.float64)
        self.bx_r = np.zeros([0,], dtype=np.float64)
        self.bz_t = np.zeros([0,], dtype=np.float64)
        self.bz_b = np.zeros([0,], dtype=np.float64)
        self.bz_half_t = np.zeros([0,], dtype=np.float64)
        self.bz_half_b = np.zeros([0,], dtype=np.float64)
        self.Psi_x_vx_l = np.zeros([0,], dtype=np.float64)
        self.Psi_x_vx_r = np.zeros([0,], dtype=np.float64)
        self.Psi_x_vz_l = np.zeros([0,], dtype=np.float64)
        self.Psi_x_vz_r = np.zeros([0,], dtype=np.float64)
        self.Psi_x_sxx_l = np.zeros([0,], dtype=np.float64)
        self.Psi_x_sxx_r = np.zeros([0,], dtype=np.float64)
        self.Psi_x_sxz_l = np.zeros([0,], dtype=np.float64)
        self.Psi_x_sxz_r = np.zeros([0,], dtype=np.float64)
        self.Psi_z_vx_t = np.zeros([0,], dtype=np.float64)
        self.Psi_z_vx_b = np.zeros([0,], dtype=np.float64)
        self.Psi_z_vz_t = np.zeros([0,], dtype=np.float64)
        self.Psi_z_vz_b = np.zeros([0,], dtype=np.float64)
        self.Psi_z_sxz_t = np.zeros([0,], dtype=np.float64)
        self.Psi_z_sxz_b = np.zeros([0,], dtype=np.float64)
        self.Psi_z_szz_t = np.zeros([0,], dtype=np.float64)
        self.Psi_z_szz_b = np.zeros([0,], dtype=np.float64)

        self.sx = sx
        self.ex = ex
        self.sz = sz
        self.ez = ez
 
    def initialize_PML_parms(self, grid: Grid_Info, verbose: int):
        
        sx = self.sx
        ex = self.ex
        sz = self.sz
        ez = self.ez
        
        in_PML = False
        Vp = 3500
        Rc = 0.00001     # PML reflection Coefficient
        N = 2

        alpha_max = grid.fpeak * np.pi
    
        # Left side

        in_PML = True
        N_left = grid.nxabs

        L = np.float64(grid.nxabs) * np.float64(grid.dl)

        d0_x = -(N + 1) * Vp * np.log(Rc)/(2.0 * L)

        if verbose > 0:
            print('\nLeft Side PML Parameters:', file=sys.stderr, flush=True)

            print('\n\tGrid Nodes {:5d}\tto\t{:5d}\t on process 0'. \
                  format(sx, N_left), \
                  file=sys.stderr, flush=True)

            print('\talpha_max: {:f}\tL: {:f}\n\td0x: {:f}'. \
                  format(alpha_max, L, d0_x), file=sys.stderr, flush=True)

        self.ax_l = np.zeros([grid.nxabs+1,], dtype=np.float64)
        self.bx_l = np.zeros([grid.nxabs+1,], dtype=np.float64)

        self.ax_half_l = np.zeros([grid.nxabs+1,], dtype=np.float64)
        self.bx_half_l = np.zeros([grid.nxabs+1,], dtype=np.float64)

        self.Psi_x_vx_l = np.zeros([grid.nxabs+1, ez - sz + 1], \
                                   dtype=np.float64)
        self.Psi_x_vz_l = np.zeros([grid.nxabs+1, ez - sz + 1], \
                                   dtype=np.float64)
        self.Psi_x_sxx_l = np.zeros([grid.nxabs+1, ez - sz + 1], \
                                    dtype=np.float64)
        self.Psi_x_sxz_l = np.zeros([grid.nxabs+1, ez - sz + 1], \
                                    dtype=np.float64)
    
        for i in range(sx, grid.nxabs+1):
 
            xi = np.float64(grid.nxabs - i) * grid.dl

            xi_L = xi/L

            dxi = d0_x * np.power (xi_L, N)

            alphaxi = alpha_max * (1.0 - xi_L)

            xi_half = (np.float64(grid.nxabs - i) - 0.5) * grid.dl

            xi_L_half = xi_half/L

            dxi_half = d0_x * np.power (xi_L_half, N)

            alphaxi_half = alpha_max * (1.0 - xi_L_half)

            kxi = 1.0

            self.bx_l[i] = np.exp (-(dxi/kxi + alphaxi)*grid.dt)
            self.ax_l[i] = dxi * (self.bx_l[i] - 1) / \
                (kxi * (dxi + kxi * alphaxi))

            if xi_half >=0:
                self.bx_half_l[i] = np.exp (-(dxi_half/kxi + \
                                              alphaxi_half)*grid.dt)
                self.ax_half_l[i] = dxi_half * (self.bx_half_l[i] - 1) / \
                (kxi * (dxi_half + kxi * alphaxi_half))
            else:
                self.bx_half_l[i] = 0
                self.ax_half_l[i] = 0

            for j in range(ez-sz):
                self.Psi_x_vx_l[i][j] = 0
                self.Psi_x_vz_l[i][j] = 0
                self.Psi_x_sxx_l[i][j] = 0
                self.Psi_x_sxz_l[i][j] = 0

        if verbose > 0:
            print('\n\tFrom Process 0', file=sys.stderr, flush=True)
            for i in range(sx, N_left+1):
                print('\tx[{:3d}]: {:.1f}\tx[{:3d}]/L: {:.2f}\tbx[{:3d}]:'. \
                      format(i, float(grid.nxabs - i) * grid.dl, i, \
                             float(grid.nxabs - i)*grid.dl/L, i), \
                      file=sys.stderr, flush=True, end='')
                print(' {:4.4g}\t  ax[{:d}]: {:4.4g}'. \
                      format(self.bx_l[i], i, self.ax_l[i]), \
                      file=sys.stderr, flush=True)

    
        # Right side

        in_PML = True
        N_right = grid.nxtot - 1
        sx_right = grid.nxtot - grid.nxabs - 1
    
        L = grid.nxabs * grid.dl

        d0_x = -(N + 1) * Vp * np.log(Rc)/(2.0 * L)

        self.ax_r = np.zeros([grid.nxabs+1,], dtype=np.float64)
        self.bx_r = np.zeros([grid.nxabs+1,], dtype=np.float64)

        self.ax_half_r = np.zeros([grid.nxabs+1,], dtype=np.float64)
        self.bx_half_r = np.zeros([grid.nxabs+1,], dtype=np.float64)

        self.Psi_x_vx_r = np.zeros([grid.nxabs+1,ez - sz + 1], \
                                   dtype=np.float64)
        self.Psi_x_vz_r = np.zeros([grid.nxabs+1,ez - sz + 1], \
                                   dtype=np.float64)
        self.Psi_x_sxx_r = np.zeros([grid.nxabs+1,ez - sz + 1], \
                                    dtype=np.float64)
        self.Psi_x_sxz_r = np.zeros([grid.nxabs+1,ez - sz + 1], \
                                    dtype=np.float64)
        if verbose > 0:
            print('\nRight Side PML Parameters:', file=sys.stderr, flush=True)

            print('\n\tGrid Nodes {:5d}\tto\t{:5d}\t on process 0'. \
                  format(sx_right, N_right), \
                  file=sys.stderr, flush=True)

            print('\talpha_max: {:f}\tL: {:f}\n\td0x: {:f}'. \
                  format(alpha_max, L, d0_x), file=sys.stderr, flush=True)

        for i in range(sx_right, N_right + 1):
            ic = i - sx_right

            xi = np.float64(i - (grid.nxtot - grid.nxabs - 1)) * grid.dl

            xi_L = xi/L

            dxi = d0_x * np.power (xi_L, N)

            alphaxi = alpha_max * (1.0 - xi_L)

            kxi = 1.0

            xi_half = (np.float64(i - (grid.nxtot - grid.nxabs - 1)) + \
                       0.5) * grid.dl

            xi_L_half = xi_half/L

            dxi_half = d0_x * np.power (xi_L_half, N)

            alphaxi_half = alpha_max * (1.0 - xi_L_half)

            self.bx_r[ic] = np.exp (-(dxi/kxi + alphaxi)*grid.dt)

            self.ax_r[ic] = dxi * (self.bx_r[ic] - 1) / \
                (kxi * (dxi + kxi * alphaxi))

            self.bx_half_r[ic] = np.exp (-(dxi_half/kxi + alphaxi_half)*grid.dt)

            self.ax_half_r[ic] = dxi_half * (self.bx_half_r[ic] - 1) / \
                (kxi * (dxi_half + kxi * alphaxi_half))

            for j in range(grid.nztot):

                self.Psi_x_vx_r[ic][j] = 0
                self.Psi_x_vz_r[ic][j] = 0
                self.Psi_x_sxx_r[ic][j] = 0
                self.Psi_x_sxz_r[ic][j] = 0

        if verbose > 0:
            print('\n\tFrom Process 0', file=sys.stderr, flush=True)
            for i in range(sx_right, N_right + 1):
                ic = i - sx_right
                print('\tx[{:3d}]: {:.1f}\tx[{:3d}]/L: {:.2f}\tbx[{:3d}]:'. \
                      format(i, float(i - (grid.nxtot - grid.nxabs - 1)) * \
                             grid.dl, i, \
                             float(i - (grid.nxtot - grid.nxabs - 1))*grid.dl/L, i), \
                    file=sys.stderr, flush=True, end='')
                print(' {:4.4g}\t  ax[{:d}]: {:4.4g}'. \
                      format(self.bx_r[ic], i, self.ax_r[ic]), \
                      file=sys.stderr, flush=True)

        # Top side

        N_top = grid.nzabs

        L = grid.nzabs * grid.dl

        d0_z = -(N + 1) * Vp * np.log(Rc)/(2.0 * L)

        self.az_t = np.zeros([grid.nzabs+1,], dtype=np.float64)
        self.bz_t = np.zeros([grid.nzabs+1,], dtype=np.float64)
 
        self.az_half_t = np.zeros([grid.nzabs+1,], dtype=np.float64)
        self.bz_half_t = np.zeros([grid.nzabs+1,], dtype=np.float64)

        self.Psi_z_vx_t = np.zeros([ex - sx + 1, grid.nzabs + 1])
        self.Psi_z_vz_t = np.zeros([ex - sx + 1, grid.nzabs + 1])
        self.Psi_z_sxz_t = np.zeros([ex - sx + 1, grid.nzabs + 1])
        self.Psi_z_szz_t = np.zeros([ex - sx + 1, grid.nzabs + 1])

        if verbose > 0:
            print('\nTop Side PML Parameters:', file=sys.stderr, flush=True)

            print('\n\tGrid Nodes {:5d}\tto\t{:5d}\t on process 0'. \
                  format(sz, N_top), \
                  file=sys.stderr, flush=True)

            print('\talpha_max: {:f}\tL: {:f}\n\td0z: {:f}'. \
                  format(alpha_max, L, d0_z), file=sys.stderr, flush=True)

        for j in range(sz, grid.nzabs + 1):

            zj = np.float64(grid.nzabs - j) * grid.dl

            zj_L = zj/L

            dzj = d0_z * np.power (zj_L, N)

            alphazj = alpha_max * (1.0 - zj_L)

            zj_half = (np.float64(grid.nzabs - j) - 0.5) * grid.dl

            zj_L_half = zj_half/L

            dzj_half = d0_z * np.power (zj_L_half, N)

            alphazj_half = alpha_max * (1.0 - zj_L_half)

            kzj = 1.0

            self.bz_t[j] = np.exp (-(dzj/kzj + alphazj)*grid.dt)

            self.az_t[j] = dzj * (self.bz_t[j] - 1) / \
                (kzj * (dzj + kzj * alphazj))

            if zj_half>=0:
                self.bz_half_t[j] = np.exp(-(dzj_half/kzj + \
                                             alphazj_half)*grid.dt)
                self.az_half_t[j] = dzj_half * (self.bz_half_t[j] - 1) / \
                    (kzj * (dzj_half + kzj * alphazj_half))
            else:
                self.bz_half_t[j] = 0
                self.az_half_t[j] = 0

            for i in range(grid.nxtot):
                self.Psi_z_vx_t[i][j] = 0
                self.Psi_z_vz_t[i][j] = 0
                self.Psi_z_sxz_t[i][j] = 0
                self.Psi_z_szz_t[i][j] = 0

            if verbose > 0:
                print('\tz[{:3d}]: {:.1f}\tz[{:3d}]/L: {:.2f}\tbz[{:3d}]:'. \
                      format(j, zj, j, zj_L, j), \
                      file=sys.stderr, flush=True, end='')
                print(' {:4.4g}\t  ax[{:d}]: {:4.4g}'. \
                      format(self.bz_t[j], j, self.az_t[j]), \
                      file=sys.stderr, flush=True)

        # Bottom side

        in_PML = True
        N_bottom = grid.nztot - 1
        sz_bottom = grid.nztot - grid.nzabs - 1
    
        L = grid.nzabs * grid.dl

        d0_z = -(N + 1) * Vp * np.log(Rc)/(2.0 * L)

        self.az_b = np.zeros([grid.nzabs+1,], dtype=np.float64)
        self.bz_b = np.zeros([grid.nzabs+1,], dtype=np.float64)

        self.az_half_b = np.zeros([grid.nzabs+1,], dtype=np.float64)
        self.bz_half_b = np.zeros([grid.nzabs+1,], dtype=np.float64)

        self.Psi_z_vx_b = np.zeros([ex - sx + 1, grid.nzabs + 1])
        self.Psi_z_vz_b = np.zeros([ex - sx + 1, grid.nzabs + 1])
        self.Psi_z_sxz_b = np.zeros([ex - sx + 1, grid.nzabs + 1])
        self.Psi_z_szz_b = np.zeros([ex - sx + 1, grid.nzabs + 1])

        if verbose > 0:
            print('\nBottom Side PML Parameters:', file=sys.stderr, flush=True)

            print('\n\tGrid Nodes {:5d}\tto\t{:5d}\t on process 0'. \
                  format(sz_bottom, N_bottom), \
                  file=sys.stderr, flush=True)
  
            print('\talpha_max: {:f}\tL: {:f}\n\td0z: {:f}'. \
                  format(alpha_max, L, d0_z), file=sys.stderr, flush=True)

        for j in range(sz_bottom, N_bottom + 1):
            jc = j - sz_bottom

            zj = np.float64(j - (grid.nztot - grid.nzabs - 1)) * grid.dl

            zj_L = zj/L

            dzj = d0_z * np.power (zj_L, N)

            alphazj = alpha_max * (1.0 - zj_L)

            zj_half = (np.float64(j - (grid.nztot - grid.nzabs - 1)) + 0.5) * \
                grid.dl

            zj_L_half = zj_half/L

            dzj_half = d0_z * np.power (zj_L_half, N)

            alphazj_half = alpha_max * (1.0 - zj_L_half)

            kzj = 1.0

            self.bz_b[jc] = np.exp (-(dzj/kzj + alphazj)*grid.dt)
            self.az_b[jc] = dzj * (self.bz_b[jc] - 1) / \
                (kzj * (dzj + kzj * alphazj))

            self.bz_half_b[jc] = np.exp (-(dzj_half/kzj + alphazj_half)*grid.dt)
            self.az_half_b[jc] = dzj_half * (self.bz_half_b[jc] - 1) / \
                (kzj * (dzj_half + kzj * alphazj_half))

            for i in range(grid.nxtot):
                self.Psi_z_vx_b[i][jc] = 0
                self.Psi_z_vz_b[i][jc] = 0
                self.Psi_z_sxz_b[i][jc] = 0
                self.Psi_z_szz_b[i][jc] = 0

            if verbose > 0:
                print('\tz[{:3d}]: {:.1f}\tz[{:3d}]/L: {:.2f}\tbz[{:3d}]:'. \
                      format(j, zj, j, zj_L, j), \
                      file=sys.stderr, flush=True, end='')
                print(' {:4.4g}\t  az[{:d}]: {:4.4g}'. \
                      format(self.bz_b[jc], j, self.az_b[jc]), \
                      file=sys.stderr, flush=True)
            


#--------------- End of function initialize_PML_parms --------------------
       
#--------------- End of class PML_Parms --------------------

class Body_Force:
    def __init__(self, grid: Grid_Info):
        
        self.isource = np.zeros([grid.nsource, grid.nwav], dtype=int)
        self.jsource = np.zeros([grid.nsource, grid.nwav], dtype=int)
        self.Fx = np.zeros([grid.nsource, grid.nwav], dtype=np.float64)
        self.Fz = np.zeros([grid.nsource, grid.nwav], dtype=np.float64)
        self.Fxz = np.zeros([grid.nsource, grid.nwav], dtype=np.float64)

        self.grid = grid

    def get_body_force(self, sx: int, ex: int, sz: int, ez: int, \
                   Cij: Stiffness):

        grid = self.grid

        if grid.wavetype == 2:
            wavelet = rickerMIT_wavelet(grid.nwav, grid.dt, grid.fpeak)
        elif grid.wavetype == 1:
            wavelet = akb_wavelet (grid.nwav, grid.dt, grid.fpeak)
        else:
            wavelet = ricker1_wavelet (grid.nwav, grid.dt, grid.fpeak)

        for i in range(grid.nsource):
            if sx <= grid.isource[i] and ex > grid.isource[i] and \
               sz <= grid.jsource[i] and ez > grid.jsource[i]:

                if grid.source_type == 'v' or grid.source_type == 'f':
                    fact = grid.dt / \
                        Cij.rho[grid.isource[i]][grid.jsource[i]]
                else:
                    fact = 1.0
            else:
                fact = 0.0

            for j in range(grid.nwav):
                self.isource[i][j] = grid.isource[i]
                self.jsource[i][j] = grid.jsource[i]
                if grid.spol_x[i] == 0.0:
                    self.Fx[i][j] = 0.0
                else:
                    self.Fx[i][j] = fact * grid.spol_x[i] * wavelet[j]
                if grid.spol_z[i] == 0.0:
                    self.Fz[i][j] = 0.0
                else:
                    self.Fz[i][j] = fact * grid.spol_z[i] * wavelet[j]
                if grid.spol_xz[i] == 0.0:
                    self.Fxz[i][j] = 0.0
                else:
                    self.Fxz[i][j] = fact * grid.spol_xz[i] * wavelet[j]

    #--------------- End of function get_body_force --------------------

#-------------------- End of class Body_Force --------------------------

def open_trace_files(tracedir: str, verbose: int):

    Vxfile = tracedir + '/Vx.bin'
    Vzfile = tracedir + '/Vz.bin'
    Pfile = tracedir + '/P.bin'

    if verbose > 0:
        print('\n\tOpening file {:s}'.format(Vxfile), \
              file=sys.stderr, flush=True)
        print('\n\tOpening file {:s}'.format(Vzfile), \
              file=sys.stderr, flush=True)
        print('\n\tOpening file {:s}'.format(Pfile), \
              file=sys.stderr, flush=True)

    Vxp = open(Vxfile, 'wb')
    Vzp = open(Vzfile, 'wb')
    Pp = open(Pfile, 'wb')

    return Vxp, Vzp, Pp

#--------------- End of function open_trace_files --------------------

def close_trace_files(Vxp: io.BufferedWriter, Vzp: io.BufferedWriter, \
                      Pp: io.BufferedWriter):

    print('\n\tClosing file {:s}'.format(Vxp.name), file=sys.stderr, flush=True)
    Vxp.flush()
    print('\t\tFlushed file {:s}'.format(Vxp.name), file=sys.stderr, flush=True)
    Vxp.close()
    print('\t\tClosed file {:s}'.format(Vxp.name), file=sys.stderr, flush=True)
    
    print('\n\tClosing file {:s}'.format(Vzp.name), file=sys.stderr, flush=True)
    Vzp.flush()
    print('\t\tFlushed file {:s}'.format(Vzp.name), file=sys.stderr, flush=True)
    Vzp.close()
    print('\t\tClosed file {:s}'.format(Vzp.name), file=sys.stderr, flush=True)
    
    print('\n\tClosing file {:s}'.format(Pp.name), file=sys.stderr, flush=True)
    Pp.flush()
    print('\t\tFlushed file {:s}'.format(Pp.name), file=sys.stderr, flush=True)
    Pp.close()
    print('\t\tClosed file {:s}'.format(Pp.name), file=sys.stderr, flush=True)

#--------------- End of function close_trace_files --------------------


def write_traces(sx: int, ex: int, sz: int, ez: int, \
                 grid: Grid_Info, vx: np.ndarray, vz: np.ndarray, \
                 sxx: np.ndarray, sxz: np.ndarray, szz: np.ndarray, \
                 Vxp: io.BufferedWriter, Vzp: io.BufferedWriter, \
                 Pp: io.BufferedWriter):

    for i in range(grid.nrec):
        vx_out = np.float32(vx[grid.i_rec[i]+2][grid.j_rec[i]+2])
        vz_out = np.float32(vz[grid.i_rec[i]+2][grid.j_rec[i]+2])
        p_out = np.float32(0.5 *(sxx[grid.i_rec[i]+2][grid.j_rec[i]+2] + \
                           szz[grid.i_rec[i]+2][grid.j_rec[i]+2]))

        Vxp.write(vx_out)
        Vzp.write(vz_out)
        Pp.write(p_out)

#--------------- End of function write_traces --------------------

def write_snapshot(it: int, sx: int, ex: int, sz: int, ez: int, \
                   grid: Grid_Info, verbose: int, snapdir: str, \
                   vx: np.ndarray, vz: np.ndarray, \
                   sxx: np.ndarray, sxz: np.ndarray, szz: np.ndarray):

    # MPI related stuff

    numprocs = 1
    myid = 0

    # Open files

    Vxfile = snapdir + '/Vx-{:07d}.bin'.format(it)
    Vzfile = snapdir + '/Vz-{:07d}.bin'.format(it)
    Pfile = snapdir + '/P-{:07d}.bin'.format(it)

    if verbose > 0:
        print('\n\tOpening file {:s}'.format(Vxfile), \
              file=sys.stderr, flush=True)
    Vxp = open(Vxfile, 'wb')
    
    if verbose > 0:
        print('\n\tOpening file {:s}'.format(Vxfile), \
              file=sys.stderr, flush=True)
    Vzp = open(Vzfile, 'wb')
    
    if verbose > 0:
        print('\n\tOpening file {:s}'.format(Pfile), \
              file=sys.stderr, flush=True)
    Pp = open(Pfile, 'wb')

    if verbose > 0:
        print('\n\tWriting file {:s}'.format(Vxfile), \
              file=sys.stderr, flush=True)
        
        print('\tWriting file {:s}'.format(Vzfile), \
              file=sys.stderr, flush=True)
        
        print('\tWriting file {:s}'.format(Pfile), \
              file=sys.stderr, flush=True)

    file_l = ez - sz

    val_x = np.zeros([ez-sz,], dtype=np.float32)
    val_z = np.zeros([ez-sz,], dtype=np.float32)
    val_p = np.zeros([ez-sz,], dtype=np.float32)

    for i in range(sx, ex):
        ic = i+2
        file_s = (grid.nztot * i + sz) * 4
        for j in range(sz, ez):
            jc = j+2
            val_x[j] = np.float32(vx[ic][jc])
            val_z[j] = np.float32(vz[ic][jc])
            val_p[j] =  np.float32(0.5*(sxx[ic][jc]+szz[ic][jc]))

        Vxp.seek(file_s, os.SEEK_SET)
        Vxp.write(val_x)
        
        Vzp.seek(file_s, os.SEEK_SET)
        Vzp.write(val_z)
        
        Pp.seek(file_s, os.SEEK_SET)
        Pp.write(val_p)

    if verbose > 0:
        print('\tClosing file {:s}'.format(Vxfile), file=sys.stderr, flush=True)
    Vxp.flush()
    Vxp.close()

    if verbose > 0:
        print('\tClosing file {:s}'.format(Vzfile), file=sys.stderr, flush=True)
    Vzp.flush()
    Vzp.close()

    if verbose > 0:
        print('\tClosing file {:s}'.format(Pfile), file=sys.stderr, flush=True)
    Pp.flush()
    Pp.close()    

#--------------- End of function write_snapshot --------------------


@jit(nopython=True)
def update_velocities(sx: int, ex: int, sz: int, ez: int, \
                      rho: np.ndarray, nxtot: int, nztot: int, \
                      nxabs: int, nzabs: int, \
                      dl: np.float64, dt: np.float64, \
                      sxx: np.ndarray, \
                      sxz: np.ndarray, szz: np.ndarray,  vx: np.ndarray, \
                      vz: np.ndarray):

    c0 = np.float64(1.125)
    c1 = np.float64(-1.0/24.0)

    #ipml = nxtot - nxabs - 1
    #jpml = nztot - nzabs - 1

    for i in range(ex-sx):
        ic = i+2
        for j in range(ez-sz):
            jc = j+2

            dsxx_dx = ( c1 * (sxx[ic+1][jc] - sxx[ic-2][jc] ) + \
                        c0 * (sxx[ic][jc] - sxx[ic-1][jc]) ) / dl

            dsxz_dz = ( c1 * (sxz[ic][jc+1] - sxz[ic][jc-2] ) + \
                        c0 * (sxz[ic][jc] - sxz[ic][jc-1]) ) / dl

            dsxz_dx = ( c1 * (sxz[ic+2][jc] - sxz[ic-1][jc] ) + \
                        c0 * (sxz[ic+1][jc] - sxz[ic][jc]) ) / dl

            dszz_dz = ( c1 * (szz[ic][jc+2] - szz[ic][jc-1] ) + \
                        c0 * (szz[ic][jc+1] - szz[ic][jc]) ) / dl

            fact = dt / rho[i][j]

            # Update velocities

            vx[ic][jc] = vx[ic][jc] + fact * (dsxz_dz + dsxx_dx)

            vz[ic][jc] = vz[ic][jc] + fact * (dsxz_dx + dszz_dz)

            if i == 0:
                vx[ic][jc] = 0.0
                vz[ic][jc] = 0.0

            if i == nxtot - 1:
                vx[ic][jc] = 0.0
                vz[ic][jc] = 0.0

            if j == 0:
                vx[ic][jc] = 0.0
                vz[ic][jc] = 0.0

            if j == nztot - 1:
                vx[ic][jc] = 0.0
                vz[ic][jc] = 0.0

    return vx, vz

#--------------- End of update_velocities --------------------


@jit(nopython=True)
def update_velocities_PML(sx: int, ex: int, sz: int, ez: int, \
                          rho: np.ndarray, nxtot: int, nztot: int, \
                          nxabs: int, nzabs: int, \
                          dl: np.float64, dt: np.float64, \
                          sxx: np.ndarray, \
                          sxz: np.ndarray, szz: np.ndarray,  \
                          ax_l: np.ndarray, \
                          ax_r: np.ndarray, \
                          ax_half_l: np.ndarray, \
                          ax_half_r: np.ndarray, \
                          az_t: np.ndarray, \
                          az_b: np.ndarray, \
                          az_half_t: np.ndarray, \
                          az_half_b: np.ndarray, \
                          bx_l: np.ndarray, \
                          bx_r: np.ndarray, \
                          bx_half_l: np.ndarray, \
                          bx_half_r: np.ndarray, \
                          bz_t: np.ndarray, \
                          bz_b: np.ndarray, \
                          bz_half_t: np.ndarray, \
                          bz_half_b: np.ndarray, \
                          Psi_x_sxx_l: np.ndarray, \
                          Psi_x_sxx_r: np.ndarray, \
                          Psi_x_sxz_l: np.ndarray, \
                          Psi_x_sxz_r: np.ndarray, \
                          Psi_z_sxz_t: np.ndarray, \
                          Psi_z_sxz_b: np.ndarray, \
                          Psi_z_szz_t: np.ndarray, \
                          Psi_z_szz_b: np.ndarray, \
                          vx: np.ndarray, \
                          vz: np.ndarray):

    c0 = np.float64(1.125)
    c1 = np.float64(-1.0/24.0)

    ipml = nxtot - nxabs - 1
    jpml = nztot - nzabs - 1

    for i in range(ex-sx):
        ic = i+2
        for j in range(ez-sz):
            jc = j+2

            dsxx_dx = ( c1 * (sxx[ic+1][jc] - sxx[ic-2][jc] ) + \
                        c0 * (sxx[ic][jc] - sxx[ic-1][jc]) ) / dl

            dsxz_dz = ( c1 * (sxz[ic][jc+1] - sxz[ic][jc-2] ) + \
                        c0 * (sxz[ic][jc] - sxz[ic][jc-1]) ) / dl

            dsxz_dx = ( c1 * (sxz[ic+2][jc] - sxz[ic-1][jc] ) + \
                        c0 * (sxz[ic+1][jc] - sxz[ic][jc]) ) / dl

            dszz_dz = ( c1 * (szz[ic][jc+2] - szz[ic][jc-1] ) + \
                        c0 * (szz[ic][jc+1] - szz[ic][jc]) ) / dl

            # Apply left side PML absorbing boundary

            if i <= nxabs:

                Psi_x_sxx_l[i][j] = bx_l[i] * Psi_x_sxx_l[i][j] + \
                    ax_l[i] * dsxx_dx

                dsxx_dx = dsxx_dx + Psi_x_sxx_l[i][j]

                Psi_x_sxz_l[i][j] = bx_half_l[i] * \
                    Psi_x_sxz_l[i][j] + ax_half_l[i] * dsxz_dx

                dsxz_dx = dsxz_dx + Psi_x_sxz_l[i][j]

            # Apply right side PML absorbing boundary

            if i >= ipml:

                ip = i - ipml

                Psi_x_sxx_r[ip][j] = bx_r[ip] * \
                    Psi_x_sxx_r[ip][j] + ax_r[ip] * dsxx_dx

                dsxx_dx = dsxx_dx + Psi_x_sxx_r[ip][j]

                Psi_x_sxz_r[ip][j] = bx_half_r[ip] * \
                    Psi_x_sxz_r[ip][j] + ax_half_r[ip] * dsxz_dx

                dsxz_dx = dsxz_dx + Psi_x_sxz_r[ip][j]

            # Apply top side PML absorbing boundary

            if j <= nzabs:

                Psi_z_szz_t[i][j] = bz_half_t[j] * \
                    Psi_z_szz_t[i][j] + az_half_t[j] * dszz_dz

                dszz_dz = dszz_dz + Psi_z_szz_t[i][j]

                Psi_z_sxz_t[i][j] = bz_t[j] * \
                    Psi_z_sxz_t[i][j] + az_t[j] * dsxz_dz

                dsxz_dz = dsxz_dz + Psi_z_sxz_t[i][j]

            # Apply bottom side PML absorbing boundary

            if j >= jpml:

                jp = j - jpml

                Psi_z_szz_b[i][jp] = bz_half_b[jp] * \
                    Psi_z_szz_b[i][jp] + az_half_b[jp] * dszz_dz

                dszz_dz = dszz_dz + Psi_z_szz_b[i][jp]

                Psi_z_sxz_b[i][jp] = bz_b[jp] * \
                    Psi_z_sxz_b[i][jp] + az_b[jp] * dsxz_dz

                dsxz_dz = dsxz_dz + Psi_z_sxz_b[i][jp]

            fact = dt / rho[i][j]

            # Update velocities

            vx[ic][jc] = vx[ic][jc] + fact * (dsxz_dz + dsxx_dx)

            vz[ic][jc] = vz[ic][jc] + fact * (dsxz_dx + dszz_dz)

            if i == 0:
                vx[ic][jc] = 0.0
                vz[ic][jc] = 0.0

            if i == nxtot - 1:
                vx[ic][jc] = 0.0
                vz[ic][jc] = 0.0

            if j == 0:
                vx[ic][jc] = 0.0
                vz[ic][jc] = 0.0

            if j == nztot - 1:
                vx[ic][jc] = 0.0
                vz[ic][jc] = 0.0

    return \
        vx, vz, \
        Psi_x_sxx_l, \
        Psi_x_sxx_r, \
        Psi_x_sxz_l, \
        Psi_x_sxz_r, \
        Psi_z_sxz_t, \
        Psi_z_sxz_b, \
        Psi_z_szz_t, \
        Psi_z_szz_b

#--------------- End of update_velocities_PML --------------------


@jit(nopython=True)
def update_stresses(sx: int, ex: int, sz: int, ez: int, \
                    c11: np.ndarray, c13: np.ndarray, c15: np.ndarray, \
                    c33: np.ndarray, c35: np.ndarray, c55: np.ndarray, \
                    rho: np.ndarray, nxtot: int, nztot: int, \
                    nxabs: int, nzabs: int, \
                    dl: np.float64, dt: np.float64, \
                    vx: np.ndarray, vz: np.ndarray, \
                    sxx: np.ndarray, sxz: np.ndarray, szz: np.ndarray):

    c0 = np.float64(1.125)
    c1 = np.float64(-1.0/24.0)

    #ipml = nxtot - nxabs - 1
    #jpml = nztot - nzabs - 1

    for i in range(ex-sx):
        ic = i+2
        for j in range(ez-sz):
            jc = j+2

            dvx_dx = ( c1 * (vx[ic+2][jc] - vx[ic-1][jc]) + \
                       c0 * (vx[ic+1][jc] - vx[ic][jc]) ) / dl

            dvx_dz = ( c1 * (vx[ic][jc+2] - vx[ic][jc-1]) + \
                       c0 * (vx[ic][jc+1] - vx[ic][jc]) ) / dl

            dvz_dx = ( c1 * (vz[ic+1][jc] - vz[ic-2][jc]) + \
                       c0 * (vz[ic][jc] - vz[ic-1][jc]) ) / dl

            dvz_dz = ( c1 * (vz[ic][jc+1] - vz[ic][jc-2]) + \
                       c0 * (vz[ic][jc] - vz[ic][jc-1]) ) / dl

            # Update stresses

            sxx[ic][jc] = sxx[ic][jc] + \
                dt * (c11[i][j] * dvx_dx + \
                           c13[i][j] * dvz_dz + \
                           c15[i][j] * (dvx_dz + dvz_dx) )

            sxz[ic][jc] = sxz[ic][jc] + \
                           dt * (c15[i][j] * dvx_dx + \
                                      c35[i][j] * dvz_dz + \
                                      c55[i][j] * (dvx_dz + dvz_dx) )

            szz[ic][jc] = szz[ic][jc] + \
                dt * (c33[i][j] * dvz_dz + \
                           c13[i][j] * dvx_dx + \
                           c35[i][j] * (dvx_dz + dvz_dx) )
                                      

    return sxx, sxz, szz

#--------------- End of update_stresses --------------------

@jit(nopython=True)
def update_stresses_PML(sx: int, ex: int, sz: int, ez: int, \
                        c11: np.ndarray, c13: np.ndarray, c15: np.ndarray, \
                        c33: np.ndarray, c35: np.ndarray, c55: np.ndarray, \
                        rho: np.ndarray, \
                        tau_sig_d: np.ndarray, tau_eps_d: np.ndarray, \
                        tau_sig_s: np.ndarray, tau_eps_s: np.ndarray, \
                        nxtot: int, nztot: int, \
                        nxabs: int, nzabs: int, \
                        dl: np.float64, dt: np.float64, \
                        vx: np.ndarray, vz: np.ndarray, \
                        ax_l: np.ndarray, \
                        ax_r: np.ndarray, \
                        ax_half_l: np.ndarray, \
                        ax_half_r: np.ndarray, \
                        az_t: np.ndarray, \
                        az_b: np.ndarray, \
                        az_half_t: np.ndarray, \
                        az_half_b: np.ndarray, \
                        bx_l: np.ndarray, \
                        bx_r: np.ndarray, \
                        bx_half_l: np.ndarray, \
                        bx_half_r: np.ndarray, \
                        bz_t: np.ndarray, \
                        bz_b: np.ndarray, \
                        bz_half_t: np.ndarray, \
                        bz_half_b: np.ndarray, \
                        Psi_x_vx_l: np.ndarray, \
                        Psi_x_vx_r: np.ndarray, \
                        Psi_x_vz_l: np.ndarray, \
                        Psi_x_vz_r: np.ndarray, \
                        Psi_z_vx_t: np.ndarray, \
                        Psi_z_vx_b: np.ndarray, \
                        Psi_z_vz_t: np.ndarray, \
                        Psi_z_vz_b: np.ndarray, \
                        eps1: np.ndarray, \
                        eps2: np.ndarray, \
                        eps3: np.ndarray, \
                        sxx: np.ndarray, sxz: np.ndarray, szz: np.ndarray):

    c0 = np.float64(1.125)
    c1 = np.float64(-1.0/24.0)

    ipml = nxtot - nxabs - 1
    jpml = nztot - nzabs - 1

    for i in range(ex-sx):
        ic = i+2
        for j in range(ez-sz):
            jc = j+2

            dvx_dx = ( c1 * (vx[ic+2][jc] - vx[ic-1][jc]) + \
                       c0 * (vx[ic+1][jc] - vx[ic][jc]) ) / dl

            dvx_dz = ( c1 * (vx[ic][jc+2] - vx[ic][jc-1]) + \
                       c0 * (vx[ic][jc+1] - vx[ic][jc]) ) / dl

            dvz_dx = ( c1 * (vz[ic+1][jc] - vz[ic-2][jc]) + \
                       c0 * (vz[ic][jc] - vz[ic-1][jc]) ) / dl

            dvz_dz = ( c1 * (vz[ic][jc+1] - vz[ic][jc-2]) + \
                       c0 * (vz[ic][jc] - vz[ic][jc-1]) ) / dl

            # Apply left side PML absorbing boundary

            if i <= nxabs:

                Psi_x_vx_l[i][j] = bx_half_l[i] * \
                    Psi_x_vx_l[i][j] + ax_half_l[i] * dvx_dx

                dvx_dx = dvx_dx + Psi_x_vx_l[i][j]

                Psi_x_vz_l[i][j] = bx_l[i] * \
                    Psi_x_vz_l[i][j] + ax_l[i] * dvz_dx

                dvz_dx = dvz_dx + Psi_x_vz_l[i][j]

            # Apply right side PML absorbing boundary

            if i >= ipml:

                ip = i - ipml

                Psi_x_vx_r[ip][j] = bx_half_r[ip] * \
                    Psi_x_vx_r[ip][j] + ax_half_r[ip] * dvx_dx

                dvx_dx = dvx_dx + Psi_x_vx_r[ip][j]

                Psi_x_vz_r[ip][j] = bx_r[ip] * \
                    Psi_x_vz_r[ip][j] + ax_r[ip] * dvz_dx

                dvz_dx = dvz_dx + Psi_x_vz_r[ip][j]

            # Apply top side PML absorbing boundary

            if j <= nzabs:

                Psi_z_vx_t[i][j] = bz_half_t[j] * \
                    Psi_z_vx_t[i][j] + az_half_t[j] * dvx_dz

                dvx_dz = dvx_dz + Psi_z_vx_t[i][j]

                Psi_z_vz_t[i][j] = bz_t[j] * \
                    Psi_z_vz_t[i][j] + az_t[j] * dvz_dz

                dvz_dz = dvz_dz + Psi_z_vz_t[i][j]

            # Apply bottom side PML absorbing boundary

            if j >= jpml:

                jp = j - jpml

                Psi_z_vx_b[i][jp] = bz_half_b[jp] * \
                    Psi_z_vx_b[i][jp] + az_half_b[jp] * dvx_dz

                dvx_dz = dvx_dz + Psi_z_vx_b[i][jp]

                Psi_z_vz_b[i][jp] = bz_b[jp] * \
                    Psi_z_vz_b[i][jp] + az_b[jp] * dvz_dz

                dvz_dz = dvz_dz + Psi_z_vz_b[i][jp]

            # Calculate Carcione, 1999 stiffness and memory variable parameters

            # Carcione, 1999 Equation (12)
            
            if tau_eps_d[i][j] < 1e-100:
                eta1 = 1.0
            else:
                eta1 = tau_sig_d[i][j] / tau_eps_d[i][j]

            if tau_eps_s[i][j] < 1e-100:
                eta2 = 1.0
            else:
                eta2 = tau_sig_s[i][j] / tau_eps_s[i][j]

            # Carcione, 1999 Equation (11)
            
            D = (c11[i][j] + c33[i][j]) / 2.0
            K = D - c55[i][j]

            # Carcione, 1999 Equations (7 - 10)

            C11_0 = c11[i][j] - D + (K * eta1) + (c55[i][j] * eta2)
            C33_0 = c33[i][j] - D + (K * eta1) + (c55[i][j] * eta2)
            C55_0 = c55[i][j] * eta2
            K0 = ( (C11_0 + C33_0) / 2.0 ) - C55_0
            

            # Update stresses with relaxed stiffness and memory variables
            # See Carcione 1999, Equation (23)

            sxx[ic][jc] = sxx[ic][jc] + \
                dt * (c11[i][j] * dvx_dx + \
                      c13[i][j] * dvz_dz + \
                      c15[i][j] * (dvx_dz + dvz_dx) + \
                      (K0 * eps1[i][j]) + \
                      (2.0 * C55_0 * eps2[i][j]) )

            sxz[ic][jc] = sxz[ic][jc] + \
                dt * (c15[i][j] * dvx_dx + \
                      c35[i][j] * dvz_dz + \
                      c55[i][j] * (dvx_dz + dvz_dx) + \
                      (C55_0 * eps3[i][j]) )

            szz[ic][jc] = szz[ic][jc] + \
                dt * (c33[i][j] * dvz_dz + \
                      c13[i][j] * dvx_dx + \
                      c35[i][j] * (dvx_dz + dvz_dx) + \
                      (K0 * eps1[i][j]) - \
                      (2.0 * C55_0 * eps2[i][j]) )

            # Update eps1[i][j] with Runge-Kutta method
            
            if tau_sig_d[i][j] > 1e-100:
                
                k1 = ( (1 - (1/eta1))*(dvx_dx + dvz_dz) - \
                       eps1[i][j] ) / \
                       tau_sig_d[i][j]
                
                k2 = ( (1 - (1/eta1))*(dvx_dx + dvz_dz) - \
                       eps1[i][j] + (dt * k1 / 2.0) ) / \
                       tau_sig_d[i][j]

                k3 = ( (1 - (1/eta1))*(dvx_dx + dvz_dz) - \
                       eps1[i][j] + (dt * k2 / 2.0) ) / \
                       tau_sig_d[i][j]

                k4 = ( (1 - (1/eta1))*(dvx_dx + dvz_dz) - \
                       eps1[i][j] + (dt * k3) ) / \
                       tau_sig_d[i][j]

                eps1[i][j] = eps1[i][j] + \
                    dt*(k1 + 2*k2 + 2*k3 + k4) / 6.0
            
            # Update eps2[i][j], eps3 with Runge-Kutta method
            
            if tau_sig_s[i][j] > 1e-100:

                # eps2[i][j]

                k1 = ( (1 - (1/eta2))*(dvx_dx - dvz_dz) - \
                       (2*eps2[i][j]) ) / \
                       (2*tau_sig_s[i][j])

                k2 = ( (1 - (1/eta2))*(dvx_dx - dvz_dz) - \
                       ( 2*(eps2[i][j] + (dt * k1 / 2.0) ) ) ) / \
                       (2*tau_sig_s[i][j])

                k3 = ( (1 - (1/eta2))*(dvx_dx - dvz_dz) - \
                       ( 2*(eps2[i][j] + (dt * k2 / 2.0) ) ) ) / \
                       (2*tau_sig_s[i][j])

                k4 = ( (1 - (1/eta2))*(dvx_dx - dvz_dz) - \
                       ( 2*(eps2[i][j] + (dt * k3) ) ) ) / \
                       (2*tau_sig_s[i][j])

                eps2[i][j] = eps2[i][j] + \
                    dt*(k1 + 2*k2 + 2*k3 + k4) / 6.0

                # eps3[i][j]

                k1 = ( (1 - (1/eta2))*(dvx_dz + dvz_dx) - \
                       eps3[i][j] ) / \
                       tau_sig_s[i][j]

                k2 = ( (1 - (1/eta2))*(dvx_dz + dvz_dx) - \
                       eps3[i][j] + (dt * k1 / 2.0) ) / \
                       tau_sig_s[i][j]

                k3 = ( (1 - (1/eta2))*(dvx_dz + dvz_dx) - \
                       eps3[i][j] + (dt * k2 / 2.0) ) / \
                       tau_sig_s[i][j]

                k4 = ( (1 - (1/eta2))*(dvx_dz + dvz_dx) - \
                       eps3[i][j] + (dt * k3) ) / \
                       tau_sig_s[i][j]

                eps3[i][j] = eps3[i][j] + \
                    dt*(k1 + 2*k2 + 2*k3 + k4) / 6.0
                                    

    return \
        sxx, sxz, szz, \
        eps1, eps2, eps3, \
        Psi_x_vx_l, \
        Psi_x_vx_r, \
        Psi_x_vz_l, \
        Psi_x_vz_r, \
        Psi_z_vx_t, \
        Psi_z_vx_b, \
        Psi_z_vz_t, \
        Psi_z_vz_b


#--------------- End of update_stresses_PML --------------------

                         
@jit(nopython=True)
def apply_body_force(it: int, sx: int, ex: int, sz: int, ez: int, \
                     dl, nsource: int, \
                     isource: np.ndarray, jsource: np.ndarray, \
                     Fx: np.ndarray, Fz: np.ndarray, Fxz: np.ndarray, \
                     vx: np.ndarray, vz: np.ndarray):

    """
    Sources applied following the body force equivalents given by:

    Robert W. Graves, "Simulating Seismic Wave Propagation in 3D Elastic Media
    Using Staggered-Grid Finite Differences",
    Bulletin of the Seismological Society of America, Vol. 86, No. 4,
    pp. 1091-1106, August 1996
    """
    
    h_3 = dl * dl *  dl

    for i in range(nsource):
        if ( (sx <= isource[i] and ex > isource[i]) and \
             (sz <= jsource[i] and ez > jsource[i]) ):
            vx[isource[i]+2 + 1][jsource[i]+2] += Fx[i][it] / h_3
            vx[isource[i]+2 - 1][jsource[i]+2] += -Fx[i][it] / h_3
            vx[isource[i]+2 - 1][jsource[i]+2 + 1] += \
                Fxz[i][it] / (4 * h_3)
            vx[isource[i]+2 + 1][jsource[i]+2 + 1] += \
                Fxz[i][it] / (4 * h_3)
            vx[isource[i]+2 - 1][jsource[i]+2 - 1] += \
                -Fxz[i][it] / (4 * h_3)
            vx[isource[i]+2 + 1][jsource[i]+2 - 1] += \
                -Fxz[i][it] / (4 * h_3)
            vz[isource[i]+2][jsource[i]+2 + 1] += Fz[i][it] / h_3
            vz[isource[i]+2][jsource[i]+2 - 1] += -Fz[i][it] / h_3
            vz[isource[i]+2 + 1][jsource[i]+2 - 1] += \
                Fxz[i][it] / (4 * h_3)
            vz[isource[i]+2 + 1][jsource[i]+2 + 1] += \
                Fxz[i][it] / (4 * h_3)
            vz[isource[i]+2 - 1][jsource[i]+2 - 1] += \
                -Fxz[i][it] / (4 * h_3)
            vz[isource[i]+2 - 1][jsource[i]+2 + 1] += \
                -Fxz[i][it] / (4 * h_3)

    return vx, vz
    

#--------------- End of apply_body_force --------------------


@jit(nopython=True)
def apply_velocity_source(it: int, sx: int, ex: int, sz: int, ez: int, \
                          nsource: int, \
                          isource: np.ndarray, jsource: np.ndarray, \
                          Fx: np.ndarray, Fz: np.ndarray, \
                          vx: np.ndarray, vz: np.ndarray):

    for i in range(nsource):
                if ( (sx <= isource[i] and ex > isource[i]) and \
             (sz <= jsource[i] and ez > jsource[i]) ):
                    vx[isource[i]+2][jsource[i]+2] += Fx[i][it]
                    vz[isource[i]+2][jsource[i]+2] += Fz[i][it]

    return vx, vz
    
#--------------- End of apply_velocity_source --------------------


@jit(nopython=True)
def apply_stress_source(it: int, sx: int, ex: int, sz: int, ez: int, \
                        nsource: int, \
                        isource: np.ndarray, jsource: np.ndarray, \
                        Fx: np.ndarray, Fz: np.ndarray, Fxz: np.ndarray, \
                        sxx: np.ndarray, sxz: np.ndarray, szz: np.ndarray):

    for i in range(nsource):
                if ( (sx <= isource[i] and ex > isource[i]) and \
             (sz <= jsource[i] and ez > jsource[i]) ):
                    sxx[isource[i]+2][jsource[i]+2] += Fx[i][it]
                    sxz[isource[i]+2][jsource[i]+2] += Fxz[i][it]
                    szz[isource[i]+2][jsource[i]+2] += Fz[i][it]
    return sxx, sxz, szz
    
#--------------- End of apply_stress_source --------------------
