import torch
import logging
import time

class DataLocation:
    def __init__(
        self, block_ids: list[int], block_size: int, used_block_lens: list[int]
    ):
        self.block_ids = block_ids
        self.block_size = block_size
        self.used_block_lens = used_block_lens

    def __repr__(self):
        return (
            f"DataLocation(block_ids={self.block_ids}, block_size={self.block_size}, "
            f"used_block_lens={self.used_block_lens})"
        )


class MemoryBlock:
    def __init__(self, block_id: int, block_size: int):
        self.block_id = block_id
        self.block_size = block_size
        self.is_free = True


class ActivationBlockCache:
    def __init__(
        self, hash_key: str, data_location: DataLocation, data_size: int, shape: int
    ):
        self.hash_key = hash_key
        self.data_location = data_location
        self.total_data_size = data_size
        self.shape = shape


class ActivationBlockPool:
    def __init__(
        self,
        block_size: int,
        device: torch.device = None,
        total_memory_size: int = 1024 * 1024 * 1024,
    ):
        self.block_size = block_size
        # 自动调整总size为block_size整数倍
        num_blocks = total_memory_size // block_size
        total_memory_size = num_blocks * block_size
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.large_buffer = torch.empty(
            total_memory_size, dtype=torch.float32, device=device
        )
        self.cached_blocks = {}  # hash_key -> ElasticBlockCache
        self.memory_blocks = [MemoryBlock(i, block_size) for i in range(num_blocks)]
        self.free_block_ids = [i for i in range(num_blocks)]

    def _find_free_blocks(self, num_blocks_needed: int) -> list[int]:
        if len(self.free_block_ids) < num_blocks_needed:
            return []
        return self.free_block_ids[:num_blocks_needed]

    def write_data(self, hash_key: str, data: torch.Tensor):
        if self.large_buffer is None:
            raise ValueError("Large buffer is not initialized.")
        if hash_key in self.cached_blocks:
            raise KeyError(f"Duplicate hash_key '{hash_key}' is not allowed.")
        data_flat = data.view(-1)
        total_len = data_flat.numel()
        num_blocks_needed = (total_len + self.block_size - 1) // self.block_size
        free_block_ids = self._find_free_blocks(num_blocks_needed)
        if len(free_block_ids) < num_blocks_needed:
            raise ValueError(
                f"Not enough free blocks. Required: {num_blocks_needed}, Available: {len(self.free_block_ids)}"
            )
        used_block_lens = []
        # 分配block并移出free_block_ids
        for i, block_id in enumerate(free_block_ids):
            memblock = self.memory_blocks[block_id]
            memblock.is_free = False
            start = i * self.block_size
            end = min(start + self.block_size, total_len)
            length = end - start
            offset = block_id * self.block_size
            self.large_buffer[offset : offset + length].copy_(data_flat[start:end])
            used_block_lens.append(length)
        self.free_block_ids = [
            bid for bid in self.free_block_ids if bid not in free_block_ids
        ]
        data_location = DataLocation(free_block_ids, self.block_size, used_block_lens)
        self.cached_blocks[hash_key] = ActivationBlockCache(
            hash_key, data_location, total_len, shape=data.shape
        )

    def read_data(self, hash_key: str) -> torch.Tensor:
        block = self.cached_blocks.get(hash_key)
        if block is not None:
            block_ids = block.data_location.block_ids
            block_size = block.data_location.block_size
            used_block_lens = block.data_location.used_block_lens
            data_pieces = []
            for i, block_id in enumerate(block_ids):
                offset = block_id * block_size
                length = used_block_lens[i]
                data_pieces.append(self.large_buffer[offset : offset + length])
            data_cat = torch.cat(data_pieces)
            if hasattr(block, "shape") and block.shape is not None:
                return data_cat.view(block.shape)
            else:
                raise ValueError("Block shape is not defined.")
        else:
            raise KeyError(f"BlockKey(hash={hash_key}) not found.")

    def free_data(self, hash_key: str):
        block_cache = self.cached_blocks.get(hash_key)
        if block_cache is None:
            raise KeyError(f"BlockKey(hash={hash_key}) not found.")
        block_ids = block_cache.data_location.block_ids
        for block_id in block_ids:
            memblock = self.memory_blocks[block_id]
            if not memblock.is_free:
                memblock.is_free = True
                self.free_block_ids.append(block_id)
        self.free_block_ids = sorted(set(self.free_block_ids))
        del self.cached_blocks[hash_key]


class StaticActivationBlockPool:
    block_size = None
    large_buffer = None
    cached_blocks = {}
    memory_blocks = []
    free_block_ids = []
    _initialized = False
    _logger = None

    def __new__(cls, *args, **kwargs):
        raise RuntimeError(
            "StaticActivationBlockPool is a static class and cannot be instantiated."
        )

    @staticmethod
    def init_pool(
        block_size: int,
        device: torch.device = None,
        total_memory_size: int = 1024 * 1024 * 1024,
    ):
        if StaticActivationBlockPool._initialized:
            StaticActivationBlockPool._logger.warning(
                "[StaticActivationBlockPool] StaticActivationBlockPool is already initialized. Re-initialization is ignored. You can re-initialize it by calling reset_pool() first."
            )
            return
        else:
            StaticActivationBlockPool.block_size = block_size
            num_blocks = total_memory_size // block_size
            total_memory_size = num_blocks * block_size
            if device is None:
                device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
            StaticActivationBlockPool.large_buffer = torch.empty(
                total_memory_size, dtype=torch.float32, device=device
            )
            StaticActivationBlockPool.cached_blocks = {}
            StaticActivationBlockPool.memory_blocks = [
                MemoryBlock(i, block_size) for i in range(num_blocks)
            ]
            StaticActivationBlockPool.free_block_ids = [i for i in range(num_blocks)]
            StaticActivationBlockPool._initialized = True
            StaticActivationBlockPool._logger = logging.getLogger(
                f"StaticActivationBlockPool at {time.time()}"
            )

    @staticmethod
    def _find_free_blocks(num_blocks_needed: int) -> list[int]:
        if len(StaticActivationBlockPool.free_block_ids) < num_blocks_needed:
            return []
        return StaticActivationBlockPool.free_block_ids[:num_blocks_needed]

    @staticmethod
    def write_data(hash_key: str, data: torch.Tensor):
        if StaticActivationBlockPool.large_buffer is None:
            raise ValueError("Large buffer is not initialized.")
        if hash_key in StaticActivationBlockPool.cached_blocks:
            raise KeyError(f"Duplicate hash_key '{hash_key}' is not allowed.")
        data_flat = data.view(-1)
        total_len = data_flat.numel()
        num_blocks_needed = (
            total_len + StaticActivationBlockPool.block_size - 1
        ) // StaticActivationBlockPool.block_size
        free_block_ids = StaticActivationBlockPool._find_free_blocks(num_blocks_needed)
        if len(free_block_ids) < num_blocks_needed:
            raise ValueError(
                f"Not enough free blocks. Required: {num_blocks_needed}, Available: {len(StaticActivationBlockPool.free_block_ids)}"
            )
        used_block_lens = []
        for i, block_id in enumerate(free_block_ids):
            memblock = StaticActivationBlockPool.memory_blocks[block_id]
            memblock.is_free = False
            start = i * StaticActivationBlockPool.block_size
            end = min(start + StaticActivationBlockPool.block_size, total_len)
            length = end - start
            offset = block_id * StaticActivationBlockPool.block_size
            StaticActivationBlockPool.large_buffer[offset : offset + length].copy_(
                data_flat[start:end]
            )
            used_block_lens.append(length)
        StaticActivationBlockPool.free_block_ids = [
            bid
            for bid in StaticActivationBlockPool.free_block_ids
            if bid not in free_block_ids
        ]
        data_location = DataLocation(
            free_block_ids, StaticActivationBlockPool.block_size, used_block_lens
        )
        StaticActivationBlockPool.cached_blocks[hash_key] = ActivationBlockCache(
            hash_key, data_location, total_len, shape=data.shape
        )
        StaticActivationBlockPool._logger.info(
            f"[StaticActivationBlockPool] Wrote data with hash_key='{hash_key}', total_len={total_len}, num_blocks_used={num_blocks_needed}."
        )

    @staticmethod
    def read_data(hash_key: str) -> torch.Tensor:
        block = StaticActivationBlockPool.cached_blocks.get(hash_key)
        if block is not None:
            block_ids = block.data_location.block_ids
            block_size = block.data_location.block_size
            used_block_lens = block.data_location.used_block_lens
            data_pieces = []
            for i, block_id in enumerate(block_ids):
                offset = block_id * block_size
                length = used_block_lens[i]
                data_pieces.append(
                    StaticActivationBlockPool.large_buffer[offset : offset + length]
                )
            data_cat = torch.cat(data_pieces)
            if hasattr(block, "shape") and block.shape is not None:
                StaticActivationBlockPool._logger.info(f"[StaticActivationBlockPool] Read data with hash_key='{hash_key}'.")
                return data_cat.view(block.shape)
            else:
                raise ValueError("Block shape is not defined.")
        else:
            raise KeyError(f"BlockKey(hash={hash_key}) not found.")

    @staticmethod
    def free_data(hash_key: str):
        block_cache = StaticActivationBlockPool.cached_blocks.get(hash_key)
        if block_cache is None:
            raise KeyError(f"BlockKey(hash={hash_key}) not found.")
        block_ids = block_cache.data_location.block_ids
        for block_id in block_ids:
            memblock = StaticActivationBlockPool.memory_blocks[block_id]
            if not memblock.is_free:
                memblock.is_free = True
                StaticActivationBlockPool.free_block_ids.append(block_id)
        StaticActivationBlockPool.free_block_ids = sorted(
            set(StaticActivationBlockPool.free_block_ids)
        )
        del StaticActivationBlockPool.cached_blocks[hash_key]
        StaticActivationBlockPool._logger.info(f"[StaticActivationBlockPool] Freed data with hash_key='{hash_key}', size={block_cache.total_data_size}.")

    @staticmethod
    def reset_pool():
        StaticActivationBlockPool.block_size = None
        StaticActivationBlockPool.large_buffer = None
        StaticActivationBlockPool.cached_blocks = {}
        StaticActivationBlockPool.memory_blocks = []
        StaticActivationBlockPool.free_block_ids = []
        StaticActivationBlockPool._initialized = False
        StaticActivationBlockPool._logger = None
