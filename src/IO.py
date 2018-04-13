import numpy as np
from numpy import empty, zeros, dot

from scipy import sparse

import os
import sys
# import time

def write_ply(path, verts, normals=None, faces=None, edges=None):
    root = os.path.dirname(__file__)

    V = verts.shape[0]
    hasnormals = normals is not None
    vertsize = 3 + 3*hasnormals

    # Fill the vert array
    vertarray = np.empty((V, vertsize), dtype=np.float32)
    vertarray[:,:3] = verts
    if hasnormals:
        vertarray[:,3:] = normals
        headpath = os.path.join(root, 'templates', 'ply_header_normals.ply')

    else:
        headpath = os.path.join(root, 'templates', 'ply_header.ply')

    with open(headpath, 'r') as headply:
        header = headply.read()

    if faces is not None:
        F = faces.shape[0]
        facesize = faces.shape[1]
        facearray = np.empty((F, 1 + 4*facesize), np.uint8)
        facearray[:,0] = facesize
        facearray[:,1:] = (faces.astype(np.int32)).view(np.uint8)
    else:
        F = 0

    if edges is not None:
        E = edges.shape[0]
        edgearray = np.empty((E, 2), np.int32)
        edgearray[:] = edges
    else:
        E = 0

    headerout = header.format(V=V, F=F, E=E)

    with open(path, 'wb') as ply:
        ply.write( bytes(headerout) )

        ply.write(vertarray.tobytes())

        if F > 0:
            ply.write(facearray.tobytes())

        if E > 0:
            ply.write(edgearray.tobytes())
