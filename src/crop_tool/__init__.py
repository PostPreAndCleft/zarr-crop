"""Cut crops out of OME-Zarr tomogram volumes without Amira.

Replaces steps 3-5 of the crop-making protocol. The source is only ever read.
"""
__all__ = ['cli', 'roi', 'zarr_io']
__version__ = '0.1.0'
