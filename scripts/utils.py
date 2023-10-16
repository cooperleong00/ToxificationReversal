from typing import Iterable, List, TypeVar

T = TypeVar('T')


def batchify(data: Iterable[T], batch_size: int, force_same=False) -> Iterable[List[T]]:
    """
    yield batches of data that all the items are the same
    """
    assert batch_size > 0

    batch = []
    if force_same:
        cur_item = data[0]
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        if force_same:
            if item == cur_item:
                batch.append(item)
            else:
                yield batch
                batch = [item]
                cur_item = item
        else:
            batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch


def repeat_interleave(data: Iterable[T], repeats: int) -> Iterable[T]:
    """
    repeat each item in data for repeats times
    """
    for item in data:
        for _ in range(repeats):
            yield item
