import torch
from vllm.v1.eTensor.activation_block import (
    ActivationBlockPool,
    StaticActivationBlockPool,
)


def test_activation_block_pool():
    def print_result(name, cond):
        print(f"{name}: {'√' if cond else '×'}")

    print("\n[ActivationBlockPool 功能测试]")
    block_size = 16
    total_size = 64  # 4 blocks
    pool = ActivationBlockPool(
        block_size=block_size, total_memory_size=total_size, device=torch.device("cpu")
    )

    # 测试1：正常写入和读取
    data1 = torch.arange(20, dtype=torch.float32)
    pool.write_data("key1", data1)
    out1 = pool.read_data("key1")
    print_result("写入/读取一致性", torch.equal(data1, out1))

    # 测试2：再次写入不同长度数据
    data2 = torch.arange(30, 50, dtype=torch.float32)
    pool.write_data("key2", data2)
    out2 = pool.read_data("key2")
    print_result("多key写入/读取一致性", torch.equal(data2, out2))

    # 测试3：重复key写入
    try:
        pool.write_data("key1", torch.ones(5, dtype=torch.float32))
        print_result("重复key写入检测", False)
    except KeyError:
        print_result("重复key写入检测", True)

    # 测试4：空间不足
    try:
        pool.write_data("key3", torch.arange(100, dtype=torch.float32))
        print_result("空间不足检测", False)
    except ValueError:
        print_result("空间不足检测", True)

    # 测试5：读取不存在key
    try:
        pool.read_data("not_exist")
        print_result("不存在key读取检测", False)
    except KeyError:
        print_result("不存在key读取检测", True)

    # 测试6：block分配和free_block_ids
    used_blocks = set()
    for k in ["key1", "key2"]:
        used_blocks.update(pool.cached_blocks[k].data_location.block_ids)
    all_blocks = set(range(total_size // block_size))
    free_blocks = set(pool.free_block_ids)
    print_result(
        "block分配正确",
        used_blocks.isdisjoint(free_blocks)
        and len(used_blocks | free_blocks) == len(all_blocks),
    )

    # 测试7：写入长度为block_size整数倍
    data3 = torch.arange(32, dtype=torch.float32)  # 2 blocks
    pool2 = ActivationBlockPool(
        block_size=16, total_memory_size=32, device=torch.device("cpu")
    )
    pool2.write_data("keyA", data3)
    out3 = pool2.read_data("keyA")
    print_result("整数倍block写入/读取一致性", torch.equal(data3, out3))

    # 测试9：写入大长度（非block_size整数倍），空间足够
    big_block_size = 128
    big_total_size = 1024  # 8 blocks
    pool_big = ActivationBlockPool(
        block_size=big_block_size,
        total_memory_size=big_total_size,
        device=torch.device("cpu"),
    )
    big_len = 1000  # 1000 < 1024, 不是block_size整数倍
    data_big = torch.arange(big_len, dtype=torch.float32)
    pool_big.write_data("bigkey", data_big)
    out_big = pool_big.read_data("bigkey")
    print_result("大长度非整数倍写入/读取一致性", torch.equal(data_big, out_big))

    # 测试10：成功删除key后无法读取
    pool_del = ActivationBlockPool(
        block_size=16, total_memory_size=32, device=torch.device("cpu")
    )
    pool_del.write_data("delkey", torch.arange(10, dtype=torch.float32))
    pool_del.free_data("delkey")
    try:
        pool_del.read_data("delkey")
        print_result("成功删除后无法读取", False)
    except KeyError:
        print_result("成功删除后无法读取", True)

    # 测试11：删除不存在的key
    try:
        pool_del.free_data("not_exist_key")
        print_result("删除不存在key检测", False)
    except KeyError:
        print_result("删除不存在key检测", True)

    # 测试12：多维tensor写入/读出，形状和内容一致
    pool_multi = ActivationBlockPool(
        block_size=32, total_memory_size=128, device=torch.device("cpu")
    )
    data_multi = torch.arange(60, dtype=torch.float32).reshape(2, 3, 2, 5)
    pool_multi.write_data("multi_key", data_multi)
    out_multi = pool_multi.read_data("multi_key")
    print_result("多维tensor形状恢复", out_multi.shape == data_multi.shape)
    print_result("多维tensor内容一致", torch.equal(out_multi, data_multi))


def test_static_activation_block_pool():
    def print_result(name, cond):
        print(f"{name}: {'√' if cond else '×'}")

    print("\n[ActivationBlockPool 功能测试]")
    block_size = 16
    total_size = 64  # 4 blocks
    StaticActivationBlockPool.init_pool(
        block_size=block_size, total_memory_size=total_size, device=torch.device("cpu")
    )

    # 测试1：正常写入和读取
    data1 = torch.arange(20, dtype=torch.float32)
    StaticActivationBlockPool.write_data("key1", data1)
    out1 = StaticActivationBlockPool.read_data("key1")
    print_result("写入/读取一致性", torch.equal(data1, out1))

    # 测试2：再次写入不同长度数据
    data2 = torch.arange(30, 50, dtype=torch.float32)
    StaticActivationBlockPool.write_data("key2", data2)
    out2 = StaticActivationBlockPool.read_data("key2")
    print_result("多key写入/读取一致性", torch.equal(data2, out2))

    # 测试3：重复key写入
    try:
        StaticActivationBlockPool.write_data("key1", torch.ones(5, dtype=torch.float32))
        print_result("重复key写入检测", False)
    except KeyError:
        print_result("重复key写入检测", True)

    # 测试4：空间不足
    try:
        StaticActivationBlockPool.write_data(
            "key3", torch.arange(100, dtype=torch.float32)
        )
        print_result("空间不足检测", False)
    except ValueError:
        print_result("空间不足检测", True)

    # 测试5：读取不存在key
    try:
        StaticActivationBlockPool.read_data("not_exist")
        print_result("不存在key读取检测", False)
    except KeyError:
        print_result("不存在key读取检测", True)

    # 测试6：block分配和free_block_ids
    used_blocks = set()
    for k in ["key1", "key2"]:
        used_blocks.update(
            StaticActivationBlockPool.cached_blocks[k].data_location.block_ids
        )
    all_blocks = set(range(total_size // block_size))
    free_blocks = set(StaticActivationBlockPool.free_block_ids)
    print_result(
        "block分配正确",
        used_blocks.isdisjoint(free_blocks)
        and len(used_blocks | free_blocks) == len(all_blocks),
    )

    # 测试7：写入长度为block_size整数倍
    StaticActivationBlockPool.init_pool(
        block_size=16, total_memory_size=32, device=torch.device("cpu")
    )
    data3 = torch.arange(32, dtype=torch.float32)  # 2 blocks
    StaticActivationBlockPool.write_data("keyA", data3)
    out3 = StaticActivationBlockPool.read_data("keyA")
    print_result("整数倍block写入/读取一致性", torch.equal(data3, out3))

    # 测试9：写入大长度（非block_size整数倍），空间足够
    StaticActivationBlockPool.init_pool(
        block_size=128, total_memory_size=1024, device=torch.device("cpu")
    )
    big_len = 1000  # 1000 < 1024, 不是block_size整数倍
    data_big = torch.arange(big_len, dtype=torch.float32)
    StaticActivationBlockPool.write_data("bigkey", data_big)
    out_big = StaticActivationBlockPool.read_data("bigkey")
    print_result("大长度非整数倍写入/读取一致性", torch.equal(data_big, out_big))

    # 测试10：成功删除key后无法读取
    StaticActivationBlockPool.init_pool(
        block_size=16, total_memory_size=32, device=torch.device("cpu")
    )
    StaticActivationBlockPool.write_data(
        "delkey", torch.arange(10, dtype=torch.float32)
    )
    StaticActivationBlockPool.free_data("delkey")
    try:
        StaticActivationBlockPool.read_data("delkey")
        print_result("成功删除后无法读取", False)
    except KeyError:
        print_result("成功删除后无法读取", True)

    # 测试11：删除不存在的key
    try:
        StaticActivationBlockPool.free_data("not_exist_key")
        print_result("删除不存在key检测", False)
    except KeyError:
        print_result("删除不存在key检测", True)

    # 测试12：多维tensor写入/读出，形状和内容一致
    StaticActivationBlockPool.init_pool(
        block_size=32, total_memory_size=128, device=torch.device("cpu")
    )
    data_multi = torch.arange(60, dtype=torch.float32).reshape(2, 3, 2, 5)
    StaticActivationBlockPool.write_data("multi_key", data_multi)
    out_multi = StaticActivationBlockPool.read_data("multi_key")
    print_result("多维tensor形状恢复", out_multi.shape == data_multi.shape)
    print_result("多维tensor内容一致", torch.equal(out_multi, data_multi))


if __name__ == "__main__":
    test_activation_block_pool()
