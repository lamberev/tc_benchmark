
"""
Performs a series of tensor network contractions, parallelizing over multiple
GPUs using cuQuantum, MPI, and NCCL.

Here we explicitly consider the following contraction:

T_{ad}^h = A_{ab}^e L_{bc}^f B_{cd}^g W^{efgh}

where superscripts have dimension dim_Fock = 30 and subscripts dimension
dim_bond = 50.

To run this script, use MPI. For example:
$ mpiexec -n 4 python contract_test/test_contractions.py
"""

import time
import cupy as cp
from cupy.cuda import nccl
from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI

from cuquantum.tensornet import Network

# --- 1. Problem Definition ---
num_contractions = 10000
dim_bond = 50
dim_Fock = 30

# Tensor contraction is T_{ad}^h = A_{ab}^e L_{bc}^f B_{cd}^g W^{efgh}
# Using a,b,c,d for bond dims and e,f,g,h for Fock dims indices.
# A_{ab}^e -> abe
# L_{bc}^f -> bcf
# B_{cd}^g -> cdg
# W^{efgh} -> efgh
# T_{ad}^h -> adh
expr = 'abe,bcf,cdg,efgh->adh'
shapes = [
    (dim_bond, dim_bond, dim_Fock),          # A
    (dim_bond, dim_bond, dim_Fock),          # L
    (dim_bond, dim_bond, dim_Fock),          # B
    (dim_Fock, dim_Fock, dim_Fock, dim_Fock) # W
]

# --- 2. MPI/NCCL Setup ---
root = 0
comm_mpi = MPI.COMM_WORLD
rank, size = comm_mpi.Get_rank(), comm_mpi.Get_size()

# Assign device for each process and set device context
device_id = rank % getDeviceCount()
cp.cuda.Device(device_id).use()

# Set up NCCL communicator
nccl_id = nccl.get_unique_id() if rank == root else None
nccl_id = comm_mpi.bcast(nccl_id, root)
comm_nccl = nccl.NcclCommunicator(size, nccl_id, rank)
stream_ptr = cp.cuda.get_current_stream().ptr

# --- 3. Pathfinding (once, before the loop) ---
# Only shapes needed to find optimal path
operands_for_path = [cp.empty(shape, dtype=cp.float64) for shape in shapes]
network_for_path = Network(expr, *operands_for_path)

# Compute the path on all ranks and specify slicing
path, info = network_for_path.contract_path(optimize={'samples': 8, 'slicing': {'min_slices': max(1, size)}})

# Select minimum cost from all ranks
opt_cost, sender = comm_mpi.allreduce(sendobj=(info.opt_cost, rank), op=MPI.MINLOC)

# Broadcast optimal path to other ranks
info = comm_mpi.bcast(info, sender)
path_info_for_contraction = {'path': info.path, 'slicing': info.slices}

# Calculate current process's share of slices
num_slices = info.num_slices
chunk, extra = num_slices // size, num_slices % size
slice_begin = rank * chunk + min(rank, extra)
slice_end = num_slices if rank == size - 1 else (rank + 1) * chunk + min(rank + 1, extra)
slices = range(slice_begin, slice_end)

if rank == root:
    print(f"Number of slices: {num_slices}")
print(f"Process {rank} is processing slice range: {slices}.")
comm_mpi.Barrier()

# --- 4. Main contraction loop ---

# Create operand tensors on all processes and initialize network + contraction
operands = [cp.empty(shape, dtype=cp.float64) for shape in shapes]
network = Network(expr, *operands)
network.contract_path(optimize=path_info_for_contraction)

if rank == root:
    start_time = time.time()

for i in range(num_contractions):
    if rank == root:
        for op in operands:
            op[...] = cp.random.rand(*op.shape, dtype=op.dtype)

    # Broadcast operands to all ranks
    for op in operands:
        comm_nccl.broadcast(op.data.ptr, op.data.ptr, op.size, nccl.NCCL_FLOAT64, root, stream_ptr)

    # Each process contracts its allocated slices
    result = network.contract(slices=slices)

    # Sum partial contributions on root
    comm_nccl.reduce(result.data.ptr, result.data.ptr, result.size, nccl.NCCL_FLOAT64, nccl.NCCL_SUM, root, stream_ptr)

# Ensure all GPU work is finished before proceeding
cp.cuda.get_current_stream().synchronize()
comm_mpi.Barrier()

# --- 5. Finalization and verification ---
if rank == root:
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time:.4f} s")
    print(f"Time per contraction: {total_time/num_contractions:.4f} s")