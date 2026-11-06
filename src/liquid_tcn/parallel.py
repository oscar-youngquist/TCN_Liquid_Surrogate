"""Bounded, ordered thread mapping for independent file I/O."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor


def ordered_io_map(function, values, workers=0):
    if workers <= 1:
        yield from map(function, values)
        return
    iterator = iter(values)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = deque()
        for _ in range(2 * workers):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass
