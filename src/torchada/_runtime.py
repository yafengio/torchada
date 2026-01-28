"""
Runtime name conversion utilities for CUDA to MUSA.

This module provides utility functions for converting CUDA function/library
names to their MUSA equivalents at runtime.

Note: torchada automatically patches ctypes.CDLL to translate function names
when loading MUSA libraries (libmusart.so, libmccl.so, etc.). Most users don't
need to use these functions directly - just import torchada and use ctypes
normally with CUDA function names.

Example of automatic patching (no code changes needed):
    import torchada
    import ctypes

    # Load MUSA runtime library
    lib = ctypes.CDLL("libmusart.so")

    # Access using CUDA function names - automatically translated!
    func = lib.cudaIpcOpenMemHandle  # -> musaIpcOpenMemHandle

These utility functions are exported for manual use if needed:
    from torchada import cuda_to_musa_name, nccl_to_mccl_name

    musa_name = cuda_to_musa_name("cudaIpcOpenMemHandle")  # -> "musaIpcOpenMemHandle"
    mccl_name = nccl_to_mccl_name("ncclAllReduce")  # -> "mcclAllReduce"
"""


def cuda_to_musa_name(name: str) -> str:
    """
    Convert a CUDA function/symbol name to its MUSA equivalent.

    This handles the common naming convention where CUDA functions start with
    "cuda" and MUSA equivalents start with "musa".

    Args:
        name: The CUDA function name (e.g., "cudaIpcOpenMemHandle")

    Returns:
        The MUSA equivalent name (e.g., "musaIpcOpenMemHandle")

    Examples:
        >>> cuda_to_musa_name("cudaMalloc")
        'musaMalloc'
        >>> cuda_to_musa_name("cudaIpcOpenMemHandle")
        'musaIpcOpenMemHandle'
        >>> cuda_to_musa_name("cudaError_t")
        'musaError_t'
        >>> cuda_to_musa_name("someOtherFunc")
        'someOtherFunc'
    """
    if name.startswith("cuda"):
        return "musa" + name[4:]
    return name


def nccl_to_mccl_name(name: str) -> str:
    """
    Convert an NCCL function/symbol name to its MCCL equivalent.

    This handles the common naming convention where NCCL functions start with
    "nccl" and MCCL equivalents start with "mccl".

    Args:
        name: The NCCL function name (e.g., "ncclAllReduce")

    Returns:
        The MCCL equivalent name (e.g., "mcclAllReduce")

    Examples:
        >>> nccl_to_mccl_name("ncclAllReduce")
        'mcclAllReduce'
        >>> nccl_to_mccl_name("ncclCommInitRank")
        'mcclCommInitRank'
        >>> nccl_to_mccl_name("ncclUniqueId")
        'mcclUniqueId'
        >>> nccl_to_mccl_name("someOtherFunc")
        'someOtherFunc'
    """
    if name.startswith("nccl"):
        return "mccl" + name[4:]
    return name


def cublas_to_mublas_name(name: str) -> str:
    """
    Convert a cuBLAS function/symbol name to its muBLAS equivalent.

    This handles the common naming convention where cuBLAS functions start with
    "cublas" and muBLAS equivalents start with "mublas".

    Args:
        name: The cuBLAS function name (e.g., "cublasCreate")

    Returns:
        The muBLAS equivalent name (e.g., "mublasCreate")

    Examples:
        >>> cublas_to_mublas_name("cublasCreate")
        'mublasCreate'
        >>> cublas_to_mublas_name("cublasSgemm")
        'mublasSgemm'
        >>> cublas_to_mublas_name("someOtherFunc")
        'someOtherFunc'
    """
    if name.startswith("cublas"):
        return "mublas" + name[6:]
    return name


def curand_to_murand_name(name: str) -> str:
    """
    Convert a cuRAND function/symbol name to its muRAND equivalent.

    Args:
        name: The cuRAND function name (e.g., "curandCreate")

    Returns:
        The muRAND equivalent name (e.g., "murandCreate")

    Examples:
        >>> curand_to_murand_name("curandCreate")
        'murandCreate'
        >>> curand_to_murand_name("curand_init")
        'murand_init'
        >>> curand_to_murand_name("someOtherFunc")
        'someOtherFunc'
    """
    if name.startswith("curand"):
        return "murand" + name[6:]
    return name
