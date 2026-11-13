# SQL and Databases Interview Notes

## Querying

### Q: [easy] What is the difference between INNER JOIN and LEFT JOIN?
An INNER JOIN returns only the rows that have matching values in both tables. A LEFT JOIN returns every row from the left table, together with matching rows from the right table; where there is no match the right-side columns are NULL. Use LEFT JOIN to keep rows that may lack related data, for example listing all customers including those with no orders, and filter on `right.id IS NULL` to find the ones with no match. RIGHT JOIN is the mirror image and FULL OUTER JOIN keeps unmatched rows from both sides.

### Q: [medium] What is the difference between GROUP BY and window functions?
GROUP BY collapses rows into one output row per group so you can compute aggregates such as COUNT, SUM or AVG. Window functions compute a value over a set of related rows but keep every original row. They use `OVER (PARTITION BY ... ORDER BY ...)`. Examples are `ROW_NUMBER()`, `RANK()`, `LAG()` and running totals with `SUM(x) OVER (ORDER BY date)`. Use window functions for rankings, top-N per group, moving averages and comparing a row with the previous one.

## Design and performance

### Q: [medium] What is an index and how does it speed up queries?
An index is an auxiliary data structure, usually a B-tree, that lets the database find rows without scanning the whole table. A B-tree keeps keys sorted in a balanced tree so lookups, range scans and ordered reads take O(log n). Indexes make reads faster but slow down inserts, updates and deletes because the index must be maintained, and they use extra storage. Index columns used in WHERE, JOIN and ORDER BY clauses; a composite index follows the leftmost-prefix rule. Low-selectivity columns, such as a boolean flag, benefit little.

### Q: [medium] Explain normalization and when to denormalize.
Normalization organises tables to remove redundancy and update anomalies. First normal form requires atomic column values, second normal form removes partial dependencies on a composite key, and third normal form removes transitive dependencies so non-key columns depend only on the key. It keeps data consistent and storage small. Denormalization deliberately duplicates data to avoid expensive joins, which speeds up read-heavy workloads such as reporting and analytics, at the cost of more complex writes and a risk of inconsistency.

### Q: [hard] How do you diagnose and optimise a slow query?
Start by running `EXPLAIN` or `EXPLAIN ANALYZE` to see the execution plan, row estimates and actual timings. Look for full table scans, poor join order, large sorts and inaccurate statistics. Typical fixes are adding or adjusting indexes, selecting only needed columns, rewriting subqueries as joins, avoiding functions on indexed columns in the WHERE clause, adding LIMIT and pagination with keyset conditions, updating table statistics, and partitioning very large tables. Measure before and after each change, and consider caching when the query result rarely changes.

## Transactions

### Q: [medium] What does ACID mean in a database?
ACID describes the guarantees of a reliable transaction. Atomicity means all operations in a transaction succeed or none do. Consistency means a transaction moves the database from one valid state to another, respecting constraints. Isolation means concurrent transactions do not see each other's partial work. Durability means once a transaction commits its changes survive crashes, normally through a write-ahead log. A classic example is a bank transfer that must debit one account and credit another as a single unit.

### Q: [hard] What are transaction isolation levels and the anomalies they prevent?
Isolation levels trade correctness against concurrency. Read uncommitted allows dirty reads, where a transaction sees uncommitted changes. Read committed prevents dirty reads but allows non-repeatable reads, where the same row changes between two reads. Repeatable read prevents that but can still allow phantom reads, where new rows appear in a repeated range query, depending on the database. Serializable prevents all anomalies by behaving as if transactions ran one after another, at the highest cost in locking or retries. Most databases default to read committed or repeatable read.
