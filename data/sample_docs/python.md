# Python Interview Notes

## Core language

### Q: [easy] What is the difference between a list and a tuple in Python?
A list is mutable: you can append, remove or reassign elements after creation. A tuple is immutable: once created, its elements cannot be replaced. Because tuples are immutable and hashable (when all their elements are hashable), they can be used as dictionary keys and set members, and they are slightly faster and more memory-efficient than lists. Use a list for a collection that changes, and a tuple for a fixed record such as a coordinate pair or a database row.

### Q: [medium] What is the Global Interpreter Lock (GIL) and how does it affect multithreading?
The GIL is a mutex in CPython that allows only one thread to execute Python bytecode at a time. It protects the interpreter's memory management, which uses reference counting, from race conditions. As a result, CPU-bound threads do not run in parallel on multiple cores, so threading gives no speed-up for heavy computation. I/O-bound programs still benefit because the GIL is released while a thread waits on the network or disk. For CPU-bound work use multiprocessing, native extensions such as NumPy that release the GIL, or a different interpreter implementation.

### Q: [medium] What are decorators and how do you write one?
A decorator is a function that takes another function and returns a new function that wraps it, adding behaviour before or after the call without modifying the original code. The `@decorator` syntax is shorthand for `func = decorator(func)`. Use `functools.wraps` so the wrapper keeps the original name and docstring.

```python
import functools, time

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.perf_counter() - start:.3f}s")
        return result
    return wrapper

@timed
def work():
    ...
```

Common uses are logging, timing, caching, authentication and retry logic.

### Q: [medium] Explain generators and the yield keyword. Why use them?
A generator function contains `yield`; calling it returns an iterator that produces values lazily, one at a time, pausing at each `yield` and resuming where it left off. Generators keep only their current state in memory, so they can process huge or even infinite sequences without building a full list. They are ideal for streaming files, pipelines of transformations, and any case where you do not need all the data at once. A generator expression such as `(x*x for x in range(10**9))` is the compact form.

### Q: [medium] Why are mutable default arguments dangerous?
Default argument values are evaluated once, when the function is defined, not each time it is called. If the default is a mutable object such as a list or dict, every call that omits the argument shares the same object, so changes leak between calls. The fix is to use `None` as the default and create a fresh object inside the function: `def add(item, bucket=None): bucket = [] if bucket is None else bucket`.

## Internals and concurrency

### Q: [hard] How does Python manage memory and garbage collection?
CPython uses reference counting as its primary mechanism: every object tracks how many references point to it, and when the count reaches zero the memory is freed immediately. Reference counting cannot reclaim reference cycles, such as two objects that point to each other, so CPython adds a generational cyclic garbage collector that periodically scans container objects for unreachable cycles. Objects are grouped into three generations; young objects are collected more often because most objects die young. Memory for small objects is handled by the pymalloc allocator, which uses pools and arenas to reduce fragmentation and system call overhead.

### Q: [hard] When should you use asyncio, threading or multiprocessing?
Use asyncio for large numbers of I/O-bound tasks, such as thousands of network requests, because a single thread cooperatively switches between coroutines at `await` points with very low overhead. Use threading for I/O-bound work that relies on blocking libraries which are not async-aware; the GIL is released during blocking I/O. Use multiprocessing, or a process pool, for CPU-bound work, because separate processes each have their own interpreter and GIL and can use multiple cores, at the cost of higher memory use and inter-process communication overhead.
