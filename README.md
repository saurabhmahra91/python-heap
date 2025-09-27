# Heap

A Python binary heap supporting both **min-heap** and **max-heap** behavior. Each node (`PQNode`) must be **unique and hashable**. The heap allows fast insert, update, remove, and pop operations.

---

## Features

- Min-heap or max-heap (`kind="min"` or `"max"`)  
- Nodes store a `key` (used for ordering) and `value` (actual data)  
- Duplicate nodes trigger a warning but are allowed  
- O(log n) insert, update, pop, and remove

---

## Installation

Copy the `Heap` class and `PQNode` protocol into your project. No external dependencies required.

---

## Usage

```python

from heap import Heap, PQNode
import warnings

class Node:
    def __init__(self, key, value):
        self.key = key       # Heap ordering
        self.value = value   # Unique data

    # Ensure uniqueness and hashability via `value`
    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return isinstance(other, Node) and self.value == other.value

# Create a min-heap
heap = Heap(kind="min")

# Insert nodes
n1 = Node(5, "apple")
n2 = Node(2, "banana")
n3 = Node(8, "cherry")

heap.insert(n1)
heap.insert(n2)
heap.insert(n3)

# Pop the top element (smallest key)
top = heap.pop()
print(top.value, top.key)  # Output: banana 2

# Update a node's key
heap.update(n3, 1)
print(heap.pop().value, heap.pop().key)  # Output: cherry 1

# Remove a node
heap.remove(n1)
