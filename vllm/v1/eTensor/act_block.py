import torch


class DataLocation:
    def __init__(self, block_ids: list[int], block_size: int, used_block_lens: list[int]):
        self.block_ids = block_ids
        self.block_size = block_size
        self.used_block_lens = used_block_lens

    def __repr__(self):
        return (f"DataLocation(block_ids={self.block_ids}, block_size={self.block_size}, "
                f"used_block_lens={self.used_block_lens})")

class MemoryBlock:
    def __init__(self, block_id: int, block_size: int):
        self.block_id = block_id
        self.block_size = block_size
        self.is_free = True

class ActivationBlockCache:
    def __init__(self, hash_key: str, data_location: DataLocation, data_size: int, shape: int):
        self.hash_key = hash_key
        self.data_location = data_location
        self.total_data_size = data_size
        self.shape = shape

class ActivationBlockPool:
    def __init__(self, 
            block_size: int, 
            device: torch.device = None, 
            total_memory_size: int = 1024 * 1024 * 1024
        ):
        self.block_size = block_size
        # 自动调整总size为block_size整数倍
        num_blocks = total_memory_size // block_size
        total_memory_size = num_blocks * block_size
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.large_buffer = torch.empty(total_memory_size, dtype=torch.float32, device=device)
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
            raise ValueError(f"Not enough free blocks. Required: {num_blocks_needed}, Available: {len(self.free_block_ids)}")
        used_block_lens = []
        # 分配block并移出free_block_ids
        for i, block_id in enumerate(free_block_ids):
            memblock = self.memory_blocks[block_id]
            memblock.is_free = False
            start = i * self.block_size
            end = min(start + self.block_size, total_len)
            length = end - start
            offset = block_id * self.block_size
            self.large_buffer[offset:offset+length].copy_(data_flat[start:end])
            used_block_lens.append(length)
        self.free_block_ids = [bid for bid in self.free_block_ids if bid not in free_block_ids]
        data_location = DataLocation(free_block_ids, self.block_size, used_block_lens)
        self.cached_blocks[hash_key] = ActivationBlockCache(hash_key, data_location, total_len, shape=data.shape)

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
                data_pieces.append(self.large_buffer[offset:offset+length])
            data_cat = torch.cat(data_pieces)
            if hasattr(block, 'shape') and block.shape is not None:
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
        
    






def test_activation_block_pool():
    def print_result(name, cond):
        print(f"{name}: {'√' if cond else '×'}")
    
    print("\n[ActivationBlockPool 功能测试]")
    block_size = 16
    total_size = 64  # 4 blocks
    pool = ActivationBlockPool(block_size=block_size, total_memory_size=total_size, device=torch.device('cpu'))

    # 测试1：正常写入和读取
    data1 = torch.arange(20, dtype=torch.float32)
    pool.write_data('key1', data1)
    out1 = pool.read_data('key1')
    print_result('写入/读取一致性', torch.equal(data1, out1))

    # 测试2：再次写入不同长度数据
    data2 = torch.arange(30, 50, dtype=torch.float32)
    pool.write_data('key2', data2)
    out2 = pool.read_data('key2')
    print_result('多key写入/读取一致性', torch.equal(data2, out2))

    # 测试3：重复key写入
    try:
        pool.write_data('key1', torch.ones(5, dtype=torch.float32))
        print_result('重复key写入检测', False)
    except KeyError:
        print_result('重复key写入检测', True)

    # 测试4：空间不足
    try:
        pool.write_data('key3', torch.arange(100, dtype=torch.float32))
        print_result('空间不足检测', False)
    except ValueError:
        print_result('空间不足检测', True)

    # 测试5：读取不存在key
    try:
        pool.read_data('not_exist')
        print_result('不存在key读取检测', False)
    except KeyError:
        print_result('不存在key读取检测', True)

    # 测试6：block分配和free_block_ids
    used_blocks = set()
    for k in ['key1', 'key2']:
        used_blocks.update(pool.cached_blocks[k].data_location.block_ids)
    all_blocks = set(range(total_size // block_size))
    free_blocks = set(pool.free_block_ids)
    print_result('block分配正确', used_blocks.isdisjoint(free_blocks) and len(used_blocks | free_blocks) == len(all_blocks))

    # 测试7：写入长度为block_size整数倍
    data3 = torch.arange(32, dtype=torch.float32)  # 2 blocks
    pool2 = ActivationBlockPool(block_size=16, total_memory_size=32, device=torch.device('cpu'))
    pool2.write_data('keyA', data3)
    out3 = pool2.read_data('keyA')
    print_result('整数倍block写入/读取一致性', torch.equal(data3, out3))


    # 测试9：写入大长度（非block_size整数倍），空间足够
    big_block_size = 128
    big_total_size = 1024  # 8 blocks
    pool_big = ActivationBlockPool(block_size=big_block_size, total_memory_size=big_total_size, device=torch.device('cpu'))
    big_len = 1000  # 1000 < 1024, 不是block_size整数倍
    data_big = torch.arange(big_len, dtype=torch.float32)
    pool_big.write_data('bigkey', data_big)
    out_big = pool_big.read_data('bigkey')
    print_result('大长度非整数倍写入/读取一致性', torch.equal(data_big, out_big))

    # 新增测试10：成功删除key后无法读取
    pool_del = ActivationBlockPool(block_size=16, total_memory_size=32, device=torch.device('cpu'))
    pool_del.write_data('delkey', torch.arange(10, dtype=torch.float32))
    pool_del.free_data('delkey')
    try:
        pool_del.read_data('delkey')
        print_result('成功删除后无法读取', False)
    except KeyError:
        print_result('成功删除后无法读取', True)

    # 新增测试11：删除不存在的key
    try:
        pool_del.free_data('not_exist_key')
        print_result('删除不存在key检测', False)
    except KeyError:
        print_result('删除不存在key检测', True)

    # 新增测试12：多维tensor写入/读出，形状和内容一致
    pool_multi = ActivationBlockPool(block_size=32, total_memory_size=128, device=torch.device('cpu'))
    data_multi = torch.arange(60, dtype=torch.float32).reshape(2, 3, 2, 5)
    pool_multi.write_data('multi_key', data_multi)
    out_multi = pool_multi.read_data('multi_key')
    print_result('多维tensor形状恢复', out_multi.shape == data_multi.shape)
    print_result('多维tensor内容一致', torch.equal(out_multi, data_multi))


if __name__ == "__main__":
    test_activation_block_pool()




