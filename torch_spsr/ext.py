import torch
from torch_spsr import _integration, _common, _sparse_op, _marching_cubes, _sparse_solve

integration = _integration
common = _common
sparse_op = _sparse_op
marching_cubes = _marching_cubes
sparse_solve = _sparse_solve

# Initialize cusolver for future usages
sparse_solve.init_cusolver()


class CuckooHashTable:
    """
    Cuckoo Hash Table for fast positional queries.
    Code is adapted from https://github.com/mit-han-lab/torchsparse
    """
    def __init__(self, data: torch.Tensor = None, hashed_data: torch.Tensor = None, enlarged: bool = False):
        """
        :param data: (N, K), K will be used as the dimension
        :param hashed_data: (N, ) if data is pre-hashed
        :param enlarged: whether or not to enlarge hash-table. Use True if there are warnings from the CUDA kernels.
        """
        if data is not None:
            self.dim = data.size(1)
            assert data.size(0) > 0
            source_hash = _common.hash_cuda(data.contiguous())
        else:
            self.dim = -1   # Never equals me.
            source_hash = hashed_data
        self.object = _common.build_hash_table(source_hash, torch.tensor([]), enlarged)

    @classmethod
    def _sphash(cls, coords: torch.Tensor, offsets=None) -> torch.Tensor:     # Int64
        """
        Compute the hash value of coords + offsets.
        :param coords: (N, 3)
        :param offsets: (K, 3) offsets to be added to coordinates
        :return: (N, K) if offsets is provided, otherwise (N,)
        """
        assert coords.dtype in [torch.int, torch.long], coords.dtype
        coords = coords.contiguous()
        if offsets is None:
            assert coords.ndim == 2 and coords.shape[1] in [2, 3, 4], coords.shape
            if coords.size(0) == 0:
                return torch.empty((coords.size(0), ), dtype=torch.int64, device=coords.device)
            return _common.hash_cuda(coords)
        else:
            assert offsets.dtype == torch.int, offsets.dtype
            assert offsets.ndim == 2 and offsets.shape[1] == 3, offsets.shape
            assert coords.ndim == 2 and coords.shape[1] in [3, 4], coords.shape
            if coords.size(0) == 0 or offsets.size(0) == 0:
                return torch.empty((offsets.size(0), coords.size(0)), dtype=torch.int64, device=coords.device)
            offsets = offsets.contiguous()
            return _common.kernel_hash_cuda(coords, offsets)

    def query(self, coords, offsets=None):
        """
        Compute the position of the queries coordinates, -1 if not found.
        :param coords: (N, 3)
        :param offsets: (K, 3)
        :return: (N, K) if offsets is provided, otherwise (N,)
        """
        assert coords.size(1) == self.dim
        hashed_query = self._sphash(coords, offsets)
        return self.query_hashed(hashed_query)

    def query_hashed(self, hashed_query: torch.Tensor):
        """
        Query the values with hashed query.
        :param hashed_query: (N, ) or (N, K)
        :return: (N, ) or (N, K)
        """
        sizes = hashed_query.size()
        hashed_query = hashed_query.view(-1)

        if hashed_query.size(0) == 0:
            return torch.zeros(sizes, dtype=torch.int64, device=hashed_query.device) - 1

        output = _common.hash_table_query(self.object, hashed_query.contiguous())
        output = (output - 1).view(*sizes)

        return output
