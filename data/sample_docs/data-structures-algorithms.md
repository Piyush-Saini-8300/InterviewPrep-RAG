# Data Structures and Algorithms Notes

## Fundamentals

### Q: [easy] What is the difference between an array and a linked list?
An array stores elements in contiguous memory, which gives O(1) random access by index and excellent cache locality, but inserting or deleting in the middle costs O(n) because elements must shift, and resizing may require copying. A linked list stores nodes connected by pointers, so insertion or deletion at a known node is O(1), but access by position is O(n) and the extra pointers use memory and hurt cache performance. Prefer arrays for lookup-heavy workloads and linked lists when you frequently insert or remove at known positions.

### Q: [medium] How does a hash table work and how are collisions handled?
A hash table applies a hash function to a key to compute an index into an array of buckets, giving average O(1) insert, lookup and delete. A collision occurs when two keys map to the same bucket. Separate chaining stores colliding entries in a linked list or tree inside the bucket. Open addressing keeps all entries in the array and probes for the next free slot using linear probing, quadratic probing or double hashing. When the load factor grows too high the table is resized and entries are rehashed to keep operations fast. Worst-case lookup is O(n) if many keys collide.

### Q: [medium] Compare breadth-first search and depth-first search.
BFS explores a graph level by level using a queue, so it finds the shortest path in an unweighted graph and is used for level-order traversal. DFS goes as deep as possible along each branch before backtracking, using recursion or a stack, and is used for cycle detection, topological sorting, connected components and exploring all possibilities in backtracking problems. Both run in O(V + E) time. BFS can use more memory on wide graphs, while DFS can use deep recursion on long graphs.

### Q: [medium] What is the time complexity of quicksort and when does it degrade?
Quicksort picks a pivot, partitions the array into elements smaller and larger than the pivot, and recursively sorts the two parts. Its average time is O(n log n) and it sorts in place with O(log n) stack space. It degrades to O(n^2) when the pivot choices are consistently poor, for example always picking the first element of an already sorted array. Choosing a random pivot or the median of three makes the worst case very unlikely. Merge sort guarantees O(n log n) and is stable, but needs O(n) extra space.

## Techniques

### Q: [medium] Explain binary search and what makes it work.
Binary search finds a target in a sorted array by repeatedly comparing it with the middle element and discarding the half that cannot contain the target, giving O(log n) time. It relies on an invariant: if the target exists, it always lies within the current low and high bounds. Common bugs are off-by-one errors in the bounds and integer overflow when computing the midpoint; compute it as `low + (high - low) // 2`. The idea extends to searching on an answer space, such as finding the minimum capacity that satisfies a condition.

### Q: [hard] What is dynamic programming and how do memoization and tabulation differ?
Dynamic programming solves problems that have overlapping subproblems and optimal substructure by solving each subproblem once and reusing its result. Memoization is top-down: write the natural recursion and cache results in a dictionary or array so repeated calls return immediately. Tabulation is bottom-up: fill a table iteratively starting from the base cases, which avoids recursion overhead and often allows space optimisation by keeping only the previous row. Classic examples are Fibonacci, the knapsack problem, longest common subsequence and edit distance. To design a solution, define the state, write the recurrence, identify base cases and decide the order of computation.

### Q: [hard] How does Dijkstra's algorithm work and what is its complexity?
Dijkstra's algorithm finds the shortest paths from a source to all other nodes in a graph with non-negative edge weights. It keeps a priority queue of nodes ordered by their best known distance. It repeatedly extracts the closest unvisited node and relaxes its outgoing edges, updating a neighbour's distance if going through the current node is shorter. With a binary heap the running time is O((V + E) log V). It fails with negative edge weights, where Bellman-Ford should be used instead.
