# System Design Interview Notes

## Scalability building blocks

### Q: [easy] What is the difference between horizontal and vertical scaling?
Vertical scaling means adding more power, such as CPU, memory or faster disks, to a single machine. It is simple but has a hard ceiling, causes downtime during upgrades and leaves a single point of failure. Horizontal scaling means adding more machines and distributing the load across them. It scales much further and improves fault tolerance, but it requires stateless services or shared state, a load balancer, and handling of distributed-systems problems such as consistency and partial failure.

### Q: [medium] What are common load balancing strategies?
A load balancer spreads requests across servers to improve throughput and availability. Round robin hands requests out in order; weighted round robin favours larger servers. Least connections sends traffic to the server with the fewest active connections, which suits long-lived requests. IP hash or consistent hashing keeps a client on the same server, giving session affinity. Layer 4 balancers route on IP and port, while layer 7 balancers can route on URL, headers or cookies. Health checks remove failed servers automatically, and the load balancer itself should be redundant.

### Q: [medium] Explain caching strategies and the cache invalidation problem.
Caching stores frequently read data in fast memory such as Redis to cut latency and database load. In cache-aside the application checks the cache first and loads from the database on a miss. Read-through and write-through put the cache in line with the data store, while write-back buffers writes and flushes them later, which is fast but risks data loss. Entries expire through a time-to-live and are evicted with policies such as LRU. Invalidation is hard because the cache can serve stale data after the source changes; remedies include short TTLs, explicit invalidation on writes and versioned keys. A cache stampede, when many requests miss at once, can be reduced with locking or request coalescing.

### Q: [medium] How do message queues help in distributed systems?
A message queue such as Kafka, RabbitMQ or SQS decouples producers from consumers. Producers publish messages and return quickly, while consumers process them at their own pace. This smooths traffic spikes, allows retries, supports asynchronous work such as sending emails or resizing images, and lets services scale and fail independently. Design for at-least-once delivery by making consumers idempotent, use dead-letter queues for messages that keep failing, and consider ordering guarantees and partitioning.

## Data and consistency

### Q: [medium] How do you choose between SQL and NoSQL databases?
Choose a relational SQL database when you need structured data, complex joins, strong consistency and ACID transactions, as in banking or order management. Choose NoSQL when you need flexible schemas, very high write throughput or horizontal scaling across many nodes. NoSQL comes in several types: key-value stores for caching and sessions, document stores for flexible records, wide-column stores for time series and large-scale writes, and graph databases for relationships. Many systems use both, picking the store that matches each access pattern.

### Q: [hard] What is the CAP theorem?
The CAP theorem says a distributed data store can provide at most two of three guarantees during a network partition: consistency, meaning every read sees the latest write; availability, meaning every request receives a response; and partition tolerance, meaning the system keeps working when messages between nodes are lost. Because partitions cannot be avoided in practice, the real choice is between consistency and availability when a partition happens. CP systems such as HBase or ZooKeeper refuse some requests to stay consistent; AP systems such as Cassandra or DynamoDB stay available and reconcile later, offering eventual consistency.

### Q: [hard] How would you design a URL shortener?
Requirements: generate a short unique code for a long URL, redirect quickly, and handle very high read traffic. The API has a create endpoint and a redirect endpoint. To generate codes, use a base62 encoding of a unique counter or ID, which avoids collisions, or a hash with collision checks. Store the mapping in a key-value or relational database indexed by code. Because reads far outnumber writes, put a cache such as Redis in front of the database and serve redirects with HTTP 301 or 302. Scale with a load balancer, stateless application servers and a sharded database, add expiry and analytics through a queue, and use rate limiting to prevent abuse.
