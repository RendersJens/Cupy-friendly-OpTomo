import numpy as np
import cupy as cp
import astra
from cupyx.scipy.sparse.linalg import LinearOperator
import operator
from functools import reduce


class OpTomo(LinearOperator):
    def __init__(self, proj_id):
        self.dtype = np.float32
        try:
            self.vol_geom = astra.projector.volume_geometry(proj_id)
            self.proj_geom = astra.projector.projection_geometry(proj_id)
            self.data_mod = astra.data2d
            self.appendString = ""
            if astra.projector.is_cuda(proj_id):
                self.appendString += "_CUDA"
        except:
            self.vol_geom = astra.projector3d.volume_geometry(proj_id)
            self.proj_geom = astra.projector3d.projection_geometry(proj_id)
            self.data_mod = astra.data3d
            self.appendString = "3D"
            if astra.projector3d.is_cuda(proj_id):
                self.appendString += "_CUDA"

        self.vol_shape = astra.functions.geom_size(self.vol_geom)
        self.vol_size = reduce(operator.mul,self.vol_shape)
        self.sino_shape = astra.functions.geom_size(self.proj_geom)
        self.sino_size = reduce(operator.mul,self.sino_shape)

        self.shape = (self.sino_size, self.vol_size)

        self.proj_id = proj_id

    def __check_array(self, arr, shape):
        if len(arr.shape)==1:
            arr = arr.reshape(shape)
        if arr.dtype != np.dtype('float32'):
            arr = arr.astype(np.dtype('float32'))
        if isinstance(arr, cp.core.core.ndarray) and arr.shape[2] % 32 != 0:
            arr_shape_2_div_32 = int(np.ceil(arr.shape[2]/32)*32)
            padded_arr = cp.empty((arr.shape[0], arr.shape[1], arr_shape_2_div_32), dtype=cp.float32)
            padded_arr[:, :, :arr.shape[2]] = arr
            arr = padded_arr[:, :, :arr.shape[2]]
        elif arr.flags['C_CONTIGUOUS']==False:
            arr = arr.copy()
        return arr

    def _matvec(self, x):
        x = self.__check_array(x, self.vol_shape)
        if isinstance(x, np.ndarray):
            # create volume id
            vol_id = self.data_mod.link('-vol', self.vol_geom, x)

            # create sino id
            out = np.empty(self.sino_shape, dtype=np.float32)
            sino_id = self.data_mod.link('-sino', self.proj_geom ,out)
        elif isinstance(x, cp.core.core.ndarray):

            # create volume id
            gpu_link_x = self.data_mod.GPULink(x.data.ptr, *x.shape[::-1], x.strides[1])
            vol_id = self.data_mod.link('-vol', self.vol_geom, gpu_link_x)

            # create sino id
            sino_shape_2_div_32 = int(np.ceil(self.sino_shape[2]/32)*32)
            out_mem = cp.empty((self.sino_shape[0], self.sino_shape[1], sino_shape_2_div_32), dtype=cp.float32)
            out = out_mem[:, :, :self.sino_shape[2]]
            gpu_link_out = self.data_mod.GPULink(out_mem.data.ptr, *out.shape[::-1], 4*sino_shape_2_div_32)
            sino_id = self.data_mod.link('-sino',self.proj_geom, gpu_link_out)
        else:
            raise ValueError("Input array should be cupy or numpy array")

        cfg = astra.creators.astra_dict('FP'+self.appendString)
        cfg['ProjectionDataId'] = sino_id
        cfg['VolumeDataId'] = vol_id
        cfg['ProjectorId'] = self.proj_id
        fp_id = astra.algorithm.create(cfg)
        astra.algorithm.run(fp_id)

        astra.algorithm.delete(fp_id)
        self.data_mod.delete([vol_id, sino_id])
        return out.ravel()

    def _transpose(self):
        return OpTomoTranspose(self.proj_id)

    # everything is real
    _adjoint = _transpose


class OpTomoTranspose(LinearOperator):
    def __init__(self, proj_id):
        self.dtype = np.float32
        try:
            self.vol_geom = astra.projector.volume_geometry(proj_id)
            self.proj_geom = astra.projector.projection_geometry(proj_id)
            self.data_mod = astra.data2d
            self.appendString = ""
            if astra.projector.is_cuda(proj_id):
                self.appendString += "_CUDA"
        except:
            self.vol_geom = astra.projector3d.volume_geometry(proj_id)
            self.proj_geom = astra.projector3d.projection_geometry(proj_id)
            self.data_mod = astra.data3d
            self.appendString = "3D"
            if astra.projector3d.is_cuda(proj_id):
                self.appendString += "_CUDA"

        self.vol_shape = astra.functions.geom_size(self.vol_geom)
        self.vol_size = reduce(operator.mul,self.vol_shape)
        self.sino_shape = astra.functions.geom_size(self.proj_geom)
        self.sino_size = reduce(operator.mul,self.sino_shape)

        self.shape = (self.vol_size, self.sino_size)

        self.proj_id = proj_id

    def __check_array(self, arr, shape):
        if len(arr.shape)==1:
            arr = arr.reshape(shape)
        if arr.dtype != np.dtype('float32'):
            arr = arr.astype(np.dtype('float32'))
        if isinstance(arr, cp.core.core.ndarray) and arr.shape[2] % 32 != 0:
            arr_shape_2_div_32 = int(np.ceil(arr.shape[2]/32)*32)
            padded_arr = cp.empty((arr.shape[0], arr.shape[1], arr_shape_2_div_32), dtype=cp.float32)
            padded_arr[:, :, :arr.shape[2]] = arr
            arr = padded_arr[:, :, :arr.shape[2]]
        elif arr.flags['C_CONTIGUOUS']==False:
            arr = arr.copy()
        return arr

    def _matvec(self, x):
        x = self.__check_array(x, self.sino_shape)
        if isinstance(x, np.ndarray):

            # create sino id
            sino_id = self.data_mod.link('-sino', self.proj_geom, x)

            # create volume id
            out = np.empty(self.vol_shape, dtype=np.float32)
            vol_id = self.data_mod.link('-vol', self.vol_geom ,out)
        elif isinstance(x, cp.core.core.ndarray):

            # create sino id
            gpu_link_x = self.data_mod.GPULink(x.data.ptr, *x.shape[::-1], x.strides[1])
            sino_id = self.data_mod.link('-sino', self.proj_geom, gpu_link_x)

            # creat volume id
            vol_shape_2_div_32 = int(np.ceil(self.vol_shape[2]/32)*32)
            out_mem = cp.empty((self.vol_shape[0], self.vol_shape[1], vol_shape_2_div_32), dtype=cp.float32)
            out = out_mem[:, :, :self.vol_shape[2]]
            gpu_link_out = self.data_mod.GPULink(out_mem.data.ptr, *out.shape[::-1], 4*vol_shape_2_div_32)
            vol_id = self.data_mod.link('-vol',self.vol_geom, gpu_link_out)
        else:
            raise ValueError("Input array should be cupy or numpy array")

        cfg = astra.creators.astra_dict('BP'+self.appendString)
        cfg['ProjectionDataId'] = sino_id
        cfg['ReconstructionDataId'] = vol_id
        cfg['ProjectorId'] = self.proj_id
        bp_id = astra.algorithm.create(cfg)
        astra.algorithm.run(bp_id)

        astra.algorithm.delete(bp_id)
        self.data_mod.delete([vol_id, sino_id])
        return out.ravel()

    def _transpose(self):
        return OpTomoTranspose(self.proj_id)

    # everything is real
    _adjoint = _transpose