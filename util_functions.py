import config as cfg
import numpy as np
import h5py
import struct
import pickle
from pathlib import Path

def loadslices_h5(fname,nslices):
    if(Path(f'{fname}.h5').exists()):
        print(f'found {fname}.h5')
        return Path(f'{fname}.h5')
    else:
        print(f'did not find {fname}.h5')
        with open(fname, 'rb') as f, h5py.File(f'{fname}.h5', 'w') as hf:
            initialized = False
            for tt in range(nslices):

                f.read(4)  # skip record marker
                IGRID = np.fromfile(f, dtype=np.int32, count=1)
                ISTEP = np.fromfile(f, dtype=np.int32, count=1)
                NJ = np.fromfile(f, dtype=np.int32, count=1)
                NK = np.fromfile(f, dtype=np.int32, count=1)
                NL = np.fromfile(f, dtype=np.int32, count=1)
                NQ = np.fromfile(f, dtype=np.int32, count=1)
                NQC = np.fromfile(f, dtype=np.int32, count=1)
                IGRID = int(np.asarray(IGRID).item());ISTEP = int(np.asarray(ISTEP).item())
                NJ = int(np.asarray(NJ).item()); NK = int(np.asarray(NK).item()); NL = int(np.asarray(NL).item())
                NQ = int(np.asarray(NQ).item()); NQC = int(np.asarray(NQC).item())
                TVREF = np.fromfile(f, dtype=np.float64, count=1)
                DTVREF = np.fromfile(f, dtype=np.float64, count=1)
                X = np.fromfile(f, dtype=np.float64, count=NJ*NK*NL)
                Y = np.fromfile(f, dtype=np.float64, count=NJ*NK*NL)
                Z = np.fromfile(f, dtype=np.float64, count=NJ*NK*NL)
                Q = np.fromfile(f, dtype=np.float64, count=NJ*NK*NL*NQ)
                IBLANK = np.fromfile(f, dtype=np.int32, count=NJ*NK*NL)
                X = X.reshape((NJ,NK,NL), order='F')
                Y = Y.reshape((NJ,NK,NL), order='F')
                Z = Z.reshape((NJ,NK,NL), order='F')
                Q = Q.reshape((NJ,NK,NL,NQ), order='F')
                IBLANK = IBLANK.reshape((NJ,NK,NL), order='F')
                f.read(4)  # skip end marker
               
                if not initialized:
                    hf.create_dataset('X',(NJ,NK,NL),dtype='f8')
                    hf.create_dataset('Y',(NJ,NK,NL),dtype='f8')
                    hf.create_dataset('Z',(NJ,NK,NL),dtype='f8')
                    hf.create_dataset('rho',(nslices,NJ,NK,NL),dtype='f8',chunks=(1,NJ,NK,NL))
                    hf.create_dataset('u',(nslices,NJ,NK,NL),dtype='f8',chunks=(1,NJ,NK,NL))
                    hf.create_dataset('v',(nslices,NJ,NK,NL),dtype='f8',chunks=(1,NJ,NK,NL))
                    hf.create_dataset('w',(nslices,NJ,NK,NL),dtype='f8',chunks=(1,NJ,NK,NL))
                    hf.create_dataset('t',(nslices,NJ,NK,NL),dtype='f8',chunks=(1,NJ,NK,NL))
                    hf.create_dataset('p',(nslices,NJ,NK,NL),dtype='f8',chunks=(1,NJ,NK,NL))   

                    hf.attrs['NJ'] = NJ
                    hf.attrs['NK'] = NK
                    hf.attrs['NL'] = NL
                    hf['X'][:,:,:] = X
                    hf['Y'][:,:,:] = Y
                    hf['Z'][:,:,:] = Z
                    initialized = True

                density = Q[:,:,:,0]*cfg.rhoinf
                velx = (Q[:,:,:,1]/Q[:,:,:,0])*cfg.ainf
                vely = (Q[:,:,:,2]/Q[:,:,:,0])*cfg.ainf
                velz = (Q[:,:,:,3]/Q[:,:,:,0])*cfg.ainf
                energy = (Q[:,:,:,4]/Q[:,:,:,0])*cfg.ainf**2
                temp = (energy - 0.5*(velx**2 + vely**2 + velz**2))/cfg.Cv
                pressure = density*cfg.R*temp

                hf['rho'][tt,:,:,:] = density
                hf['u'][tt,:,:,:]   = velx
                hf['v'][tt,:,:,:]   = vely
                hf['w'][tt,:,:,:]   = velz
                hf['t'][tt,:,:,:]   = temp
                hf['p'][tt,:,:,:]   = pressure

        return Path(f'{fname}.h5')



def loadslices(fname,nslices):
    if(Path(f"{fname}.pkl").exists()):
        with open(f"{fname}.pkl", "rb") as g:
            slices = pickle.load(g)

    else:
        slices = [None]*nslices
        with open(fname, "rb") as f:
            for ii in range(nslices):
            
               f.read(4)  # skip record marker
               IGRID = np.fromfile(f, dtype=np.int32, count=1)
               ISTEP = np.fromfile(f, dtype=np.int32, count=1)
               NJ = np.fromfile(f, dtype=np.int32, count=1)
               NK = np.fromfile(f, dtype=np.int32, count=1)
               NL = np.fromfile(f, dtype=np.int32, count=1)
               NQ = np.fromfile(f, dtype=np.int32, count=1)
               NQC = np.fromfile(f, dtype=np.int32, count=1)
               IGRID = int(np.asarray(IGRID).item());ISTEP = int(np.asarray(ISTEP).item())
               NJ = int(np.asarray(NJ).item()); NK = int(np.asarray(NK).item()); NL = int(np.asarray(NL).item())
               NQ = int(np.asarray(NQ).item()); NQC = int(np.asarray(NQC).item())
               TVREF = np.fromfile(f, dtype=np.float64, count=1)
               DTVREF = np.fromfile(f, dtype=np.float64, count=1)
               X = np.fromfile(f, dtype=np.float64, count=NJ*NK*NL)
               Y = np.fromfile(f, dtype=np.float64, count=NJ*NK*NL)
               Z = np.fromfile(f, dtype=np.float64, count=NJ*NK*NL)
               Q = np.fromfile(f, dtype=np.float64, count=NJ*NK*NL*NQ)
               IBLANK = np.fromfile(f, dtype=np.int32, count=NJ*NK*NL)
               X = X.reshape((NJ,NK,NL), order='F')
               Y = Y.reshape((NJ,NK,NL), order='F')
               Z = Z.reshape((NJ,NK,NL), order='F')
               Q = Q.reshape((NJ,NK,NL,NQ), order='F')
               IBLANK = IBLANK.reshape((NJ,NK,NL), order='F')
               f.read(4)  # skip end marker
               
               if ii == 0:
                  slices[ii] = {"IGRID":IGRID, "ISTEP":ISTEP, "NJ":NJ, "NK":NK, "NL":NL,
                                "NQ":NQ, "NQC":NQC, "TVREF":TVREF, "DTVREF":DTVREF,
                                "X":X, "Y":Y, "Z":Z, "Q":Q}
               else:
                  slices[ii] = {"IGRID":IGRID, "ISTEP":ISTEP, "NJ":NJ, "NK":NK, "NL":NL,
                                "NQ":NQ, "NQC":NQC, "TVREF":TVREF, "DTVREF":DTVREF,
                                "Q":Q}
                  

        with open(f"{fname}.pkl", "wb") as g:
            pickle.dump(slices, g, protocol=pickle.HIGHEST_PROTOCOL)

    return slices


def computeSchlieren(NJ, NL, XC, YC, RHO):
    rhoGradMag = np.zeros((NJ,1,NL))
    for JJ in range(1,NJ-1):
        for LL in range(1,NL-1):
            eps = 1e-12
            dx = np.array([XC[JJ-1,LL] - XC[JJ,LL],
                           XC[JJ+1,LL] - XC[JJ,LL],
                           XC[JJ,LL-1] - XC[JJ,LL],
                           XC[JJ,LL+1] - XC[JJ,LL]])
            dy = np.array([YC[JJ-1,LL] - YC[JJ,LL],
                           YC[JJ+1,LL] - YC[JJ,LL],
                           YC[JJ,LL-1] - YC[JJ,LL],
                           YC[JJ,LL+1] - YC[JJ,LL]])
            dr = np.array([RHO[JJ-1,LL] - RHO[JJ,LL],
                           RHO[JJ+1,LL] - RHO[JJ,LL],
                           RHO[JJ,LL-1] - RHO[JJ,LL],
                           RHO[JJ,LL+1] - RHO[JJ,LL]])
            w = 1.0 / (dx*dx + dy*dy + eps)

            A11 = np.sum(w*dx*dx)
            A12 = np.sum(w*dx*dy)
            A22 = np.sum(w*dy*dy)
            b1 = np.sum(w*dx*dr)
            b2 = np.sum(w*dy*dr)

            A = np.array([[A11, A12],[A12, A22]])
            b = np.array([b1, b2])

            try:
                grad = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                grad = np.linalg.lstsq(A, b, rcond=None)[0]

            d_rho_dx = grad[0]
            d_rho_dy = grad[1]
            rhoGradMag[JJ,0,LL] = np.sqrt(d_rho_dx**2 + d_rho_dy**2)

    return rhoGradMag
