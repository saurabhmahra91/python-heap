import unittest

from .heap import Heap


class TestNode:
    """
    Simple test node implementation for testing Heap.
    """

    def __init__(self, key: int | float, value: str = ""):
        self.key = key
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if not isinstance(other, TestNode):
            return False
        return self.value == other.value

    def __repr__(self):
        return f"TestNode(key={self.key}, value='{self.value}')"


class TestMinHeap(unittest.TestCase):
    """
    Test suite for MinHeap functionality.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        self.HeapClass = Heap

    def test_init_minheap(self):
        """
        Test min heap initialization.
        """
        heap = self.HeapClass(kind="min")
        self.assertIsNotNone(heap)
        # Heap should be empty initially
        with self.assertRaises((IndexError, KeyError, ValueError)):
            heap.pop()

    def test_init_default_is_minheap(self):
        """
        Test that default initialization creates a min heap.
        """
        heap = self.HeapClass()  # No kind specified, should default to "min"
        nodes = [TestNode(3, "three"), TestNode(1, "one"), TestNode(2, "two")]
        for node in nodes:
            heap.insert(node)

        # Should pop in ascending order (min heap behavior)
        self.assertEqual(heap.pop().key, 1)
        self.assertEqual(heap.pop().key, 2)
        self.assertEqual(heap.pop().key, 3)

    def test_insert_single_element_minheap(self):
        """
        Test inserting a single element in min heap.
        """
        heap = self.HeapClass(kind="min")
        node = TestNode(5, "five")
        heap.insert(node)
        popped = heap.pop()
        self.assertEqual(popped.key, 5)
        self.assertEqual(popped.value, "five")

    def test_insert_multiple_elements_minheap(self):
        """
        Test inserting multiple elements maintains min-heap property.
        """
        heap = self.HeapClass(kind="min")
        nodes = [
            TestNode(5, "five"),
            TestNode(3, "three"),
            TestNode(8, "eight"),
            TestNode(1, "one"),
            TestNode(4, "four"),
        ]

        for node in nodes:
            heap.insert(node)

        # Elements should come out in ascending order (min first)
        expected_keys = [1, 3, 4, 5, 8]
        for expected_key in expected_keys:
            popped = heap.pop()
            self.assertEqual(popped.key, expected_key)

    def test_insert_duplicate_keys_minheap(self):
        """
        Test inserting nodes with duplicate keys in min heap.
        """
        heap = self.HeapClass(kind="min")
        nodes = [
            TestNode(3, "three-a"),
            TestNode(3, "three-b"),
            TestNode(3, "three-c"),
            TestNode(1, "one"),
            TestNode(5, "five"),
        ]

        for node in nodes:
            heap.insert(node)

        # First should be the node with key 1
        self.assertEqual(heap.pop().key, 1)

        # Next three should all have key 3 (order among them may vary)
        for _ in range(3):
            self.assertEqual(heap.pop().key, 3)

        # Last should have key 5
        self.assertEqual(heap.pop().key, 5)

    def test_update_decrease_key_minheap(self):
        """
        Test updating a node's key to a smaller value in min heap.
        """
        heap = self.HeapClass(kind="min")
        node1 = TestNode(10, "ten")
        node2 = TestNode(5, "five")
        node3 = TestNode(15, "fifteen")

        heap.insert(node1)
        heap.insert(node2)
        heap.insert(node3)

        # Update node1's key from 10 to 2
        heap.update(node1, 2)

        # node1 should now be the minimum
        popped = heap.pop()
        self.assertEqual(popped.value, "ten")
        self.assertEqual(popped.key, 2)

    def test_update_increase_key_minheap(self):
        """
        Test updating a node's key to a larger value in min heap.
        """
        heap = self.HeapClass(kind="min")
        node1 = TestNode(3, "three")
        node2 = TestNode(7, "seven")
        node3 = TestNode(9, "nine")

        heap.insert(node1)
        heap.insert(node2)
        heap.insert(node3)

        # Update node1's key from 3 to 10
        heap.update(node1, 10)

        # node2 should now be the minimum
        self.assertEqual(heap.pop().key, 7)
        self.assertEqual(heap.pop().key, 9)
        popped = heap.pop()
        self.assertEqual(popped.value, "three")
        self.assertEqual(popped.key, 10)

    def test_update_nonexistent_node_minheap(self):
        """
        Test updating a node that's not in the heap.
        """
        heap = self.HeapClass(kind="min")
        heap.insert(TestNode(5, "five"))

        nonexistent = TestNode(10, "ten")
        with self.assertRaises((KeyError, ValueError)):
            heap.update(nonexistent, 3)

    def test_pop_empty_heap_minheap(self):
        """
        Test popping from an empty min heap.
        """
        heap = self.HeapClass(kind="min")
        with self.assertRaises((IndexError, KeyError, ValueError)):
            heap.pop()

    def test_pop_order_minheap(self):
        """
        Test that pop returns elements in correct order for min heap.
        """
        heap = self.HeapClass(kind="min")
        keys = [7, 2, 9, 4, 1, 8, 5, 3, 6]
        nodes = [TestNode(k, str(k)) for k in keys]

        for node in nodes:
            heap.insert(node)

        popped_keys = []
        for _ in range(len(keys)):
            popped_keys.append(heap.pop().key)

        self.assertEqual(popped_keys, sorted(keys))

    def test_remove_root_minheap(self):
        """
        Test removing the root element from min heap.
        """
        heap = self.HeapClass(kind="min")
        node1 = TestNode(1, "one")
        node2 = TestNode(3, "three")
        node3 = TestNode(5, "five")

        heap.insert(node1)
        heap.insert(node2)
        heap.insert(node3)

        heap.remove(node1)

        # node2 should now be the minimum
        self.assertEqual(heap.pop().key, 3)
        self.assertEqual(heap.pop().key, 5)

    def test_remove_middle_element_minheap(self):
        """
        Test removing an element from the middle of the min heap.
        """
        heap = self.HeapClass(kind="min")
        nodes = [TestNode(i, str(i)) for i in [1, 3, 5, 7, 9]]

        for node in nodes:
            heap.insert(node)

        # Remove the node with key 5
        heap.remove(nodes[2])

        # Remaining elements should still maintain heap property
        expected = [1, 3, 7, 9]
        for expected_key in expected:
            self.assertEqual(heap.pop().key, expected_key)

    def test_remove_last_element_minheap(self):
        """
        Test removing the last element from min heap.
        """
        heap = self.HeapClass(kind="min")
        node = TestNode(5, "five")
        heap.insert(node)
        heap.remove(node)

        # Heap should be empty now
        with self.assertRaises((IndexError, KeyError, ValueError)):
            heap.pop()

    def test_remove_nonexistent_node_minheap(self):
        """
        Test removing a node that's not in the min heap.
        """
        heap = self.HeapClass(kind="min")
        heap.insert(TestNode(5, "five"))

        nonexistent = TestNode(10, "ten")
        with self.assertRaises((KeyError, ValueError)):
            heap.remove(nonexistent)

    def test_float_keys_minheap(self):
        """
        Test min heap with float keys.
        """
        heap = self.HeapClass(kind="min")
        nodes = [TestNode(3.14, "pi"), TestNode(2.71, "e"), TestNode(1.41, "sqrt2"), TestNode(1.73, "sqrt3")]

        for node in nodes:
            heap.insert(node)

        expected_keys = [1.41, 1.73, 2.71, 3.14]
        for expected_key in expected_keys:
            self.assertAlmostEqual(heap.pop().key, expected_key, places=2)

    def test_negative_keys_minheap(self):
        """
        Test min heap with negative keys.
        """
        heap = self.HeapClass(kind="min")
        nodes = [TestNode(-1, "neg_one"), TestNode(5, "five"), TestNode(-10, "neg_ten"), TestNode(0, "zero")]

        for node in nodes:
            heap.insert(node)

        expected_keys = [-10, -1, 0, 5]
        for expected_key in expected_keys:
            self.assertEqual(heap.pop().key, expected_key)


class TestMaxHeap(unittest.TestCase):
    """
    Test suite for MaxHeap functionality.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """

        self.HeapClass = Heap

    def test_init_maxheap(self):
        """
        Test max heap initialization.
        """
        heap = self.HeapClass(kind="max")
        self.assertIsNotNone(heap)
        # Heap should be empty initially
        with self.assertRaises((IndexError, KeyError, ValueError)):
            heap.pop()

    def test_insert_single_element_maxheap(self):
        """
        Test inserting a single element in max heap.
        """
        heap = self.HeapClass(kind="max")
        node = TestNode(5, "five")
        heap.insert(node)
        popped = heap.pop()
        self.assertEqual(popped.key, 5)
        self.assertEqual(popped.value, "five")

    def test_insert_multiple_elements_maxheap(self):
        """
        Test inserting multiple elements maintains max-heap property.
        """
        heap = self.HeapClass(kind="max")
        nodes = [
            TestNode(5, "five"),
            TestNode(3, "three"),
            TestNode(8, "eight"),
            TestNode(1, "one"),
            TestNode(4, "four"),
        ]

        for node in nodes:
            heap.insert(node)

        # Elements should come out in descending order (max first)
        expected_keys = [8, 5, 4, 3, 1]
        for expected_key in expected_keys:
            popped = heap.pop()
            self.assertEqual(popped.key, expected_key)

    def test_insert_duplicate_keys_maxheap(self):
        """
        Test inserting nodes with duplicate keys in max heap.
        """
        heap = self.HeapClass(kind="max")
        nodes = [
            TestNode(3, "three-a"),
            TestNode(3, "three-b"),
            TestNode(3, "three-c"),
            TestNode(1, "one"),
            TestNode(5, "five"),
        ]

        for node in nodes:
            heap.insert(node)

        # First should be the node with key 5
        self.assertEqual(heap.pop().key, 5)

        # Next three should all have key 3 (order among them may vary)
        for _ in range(3):
            self.assertEqual(heap.pop().key, 3)

        # Last should have key 1
        self.assertEqual(heap.pop().key, 1)

    def test_update_increase_key_maxheap(self):
        """
        Test updating a node's key to a larger value in max heap.
        """
        heap = self.HeapClass(kind="max")
        node1 = TestNode(10, "ten")
        node2 = TestNode(5, "five")
        node3 = TestNode(15, "fifteen")

        heap.insert(node1)
        heap.insert(node2)
        heap.insert(node3)

        # Update node2's key from 5 to 20
        heap.update(node2, 20)

        # node2 should now be the maximum
        popped = heap.pop()
        self.assertEqual(popped.value, "five")
        self.assertEqual(popped.key, 20)

    def test_update_decrease_key_maxheap(self):
        """
        Test updating a node's key to a smaller value in max heap.
        """
        heap = self.HeapClass(kind="max")
        node1 = TestNode(10, "ten")
        node2 = TestNode(7, "seven")
        node3 = TestNode(5, "five")

        heap.insert(node1)
        heap.insert(node2)
        heap.insert(node3)

        # Update node1's key from 10 to 3
        heap.update(node1, 3)

        # node2 should now be the maximum
        self.assertEqual(heap.pop().key, 7)
        self.assertEqual(heap.pop().key, 5)
        popped = heap.pop()
        self.assertEqual(popped.value, "ten")
        self.assertEqual(popped.key, 3)

    def test_pop_order_maxheap(self):
        """
        Test that pop returns elements in correct order for max heap.
        """
        heap = self.HeapClass(kind="max")
        keys = [7, 2, 9, 4, 1, 8, 5, 3, 6]
        nodes = [TestNode(k, str(k)) for k in keys]

        for node in nodes:
            heap.insert(node)

        popped_keys = []
        for _ in range(len(keys)):
            popped_keys.append(heap.pop().key)

        self.assertEqual(popped_keys, sorted(keys, reverse=True))

    def test_remove_root_maxheap(self):
        """
        Test removing the root element from max heap.
        """
        heap = self.HeapClass(kind="max")
        node1 = TestNode(5, "five")
        node2 = TestNode(3, "three")
        node3 = TestNode(1, "one")

        heap.insert(node1)
        heap.insert(node2)
        heap.insert(node3)

        heap.remove(node1)  # Remove max element

        # node2 should now be the maximum
        self.assertEqual(heap.pop().key, 3)
        self.assertEqual(heap.pop().key, 1)

    def test_remove_middle_element_maxheap(self):
        """
        Test removing an element from the middle of the max heap.
        """
        heap = self.HeapClass(kind="max")
        nodes = [TestNode(i, str(i)) for i in [1, 3, 5, 7, 9]]

        for node in nodes:
            heap.insert(node)

        # Remove the node with key 5
        heap.remove(nodes[2])

        # Remaining elements should still maintain heap property
        expected = [9, 7, 3, 1]
        for expected_key in expected:
            self.assertEqual(heap.pop().key, expected_key)

    def test_float_keys_maxheap(self):
        """
        Test max heap with float keys.
        """
        heap = self.HeapClass(kind="max")
        nodes = [TestNode(3.14, "pi"), TestNode(2.71, "e"), TestNode(1.41, "sqrt2"), TestNode(1.73, "sqrt3")]

        for node in nodes:
            heap.insert(node)

        expected_keys = [3.14, 2.71, 1.73, 1.41]
        for expected_key in expected_keys:
            self.assertAlmostEqual(heap.pop().key, expected_key, places=2)

    def test_negative_keys_maxheap(self):
        """
        Test max heap with negative keys.
        """
        heap = self.HeapClass(kind="max")
        nodes = [TestNode(-1, "neg_one"), TestNode(5, "five"), TestNode(-10, "neg_ten"), TestNode(0, "zero")]

        for node in nodes:
            heap.insert(node)

        expected_keys = [5, 0, -1, -10]
        for expected_key in expected_keys:
            self.assertEqual(heap.pop().key, expected_key)


class TestHeapStress(unittest.TestCase):
    """
    Stress tests for both min and max heap.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        self.HeapClass = Heap

    def test_stress_test_many_operations_minheap(self):
        """
        Stress test with many operations on min heap.
        """
        heap = self.HeapClass(kind="min")
        nodes = []

        # Insert many elements
        for i in range(100, 0, -1):
            node = TestNode(i, f"node_{i}")
            nodes.append(node)
            heap.insert(node)

        # Update some keys
        heap.update(nodes[50], -5)  # Should become new minimum
        heap.update(nodes[25], 200)  # Should move toward end

        # Remove some nodes
        heap.remove(nodes[10])
        heap.remove(nodes[90])

        # First element should be the one we updated to -5
        first = heap.pop()
        self.assertEqual(first.key, -5)

        # Pop remaining elements and verify they're in order
        prev_key = first.key
        while True:
            try:
                current = heap.pop()
                self.assertGreaterEqual(current.key, prev_key)
                prev_key = current.key
            except (IndexError, KeyError, ValueError):
                break

    def test_stress_test_many_operations_maxheap(self):
        """
        Stress test with many operations on max heap.
        """
        heap = self.HeapClass(kind="max")
        nodes = []

        # Insert many elements
        for i in range(1, 101):
            node = TestNode(i, f"node_{i}")
            nodes.append(node)
            heap.insert(node)

        # Update some keys
        heap.update(nodes[50], 150)  # Should become new maximum
        heap.update(nodes[75], -10)  # Should move toward end

        # Remove some nodes
        heap.remove(nodes[10])
        heap.remove(nodes[90])

        # First element should be the one we updated to 150
        first = heap.pop()
        self.assertEqual(first.key, 150)

        # Pop remaining elements and verify they're in descending order
        prev_key = first.key
        while True:
            try:
                current = heap.pop()
                self.assertLessEqual(current.key, prev_key)
                prev_key = current.key
            except (IndexError, KeyError, ValueError):
                break

    def test_alternating_heap_types(self):
        """
        Test that min and max heaps work correctly when used together.
        """
        min_heap = self.HeapClass(kind="min")
        max_heap = self.HeapClass(kind="max")

        values = [5, 3, 8, 1, 9, 2, 7, 4, 6]
        min_nodes = [TestNode(v, f"min_{v}") for v in values]
        max_nodes = [TestNode(v, f"max_{v}") for v in values]

        for min_node, max_node in zip(min_nodes, max_nodes):
            min_heap.insert(min_node)
            max_heap.insert(max_node)

        # Min heap should give smallest first
        self.assertEqual(min_heap.pop().key, 1)

        # Max heap should give largest first
        self.assertEqual(max_heap.pop().key, 9)

        # Update both heaps
        min_heap.update(min_nodes[1], 0)  # 3 -> 0 in min heap
        max_heap.update(max_nodes[1], 10)  # 3 -> 10 in max heap

        # Check new roots
        self.assertEqual(min_heap.pop().key, 0)
        self.assertEqual(max_heap.pop().key, 10)

    def test_same_node_multiple_operations_minheap(self):
        """
        Test multiple operations on the same node in min heap.
        """
        heap = self.HeapClass(kind="min")
        node1 = TestNode(5, "five")
        node2 = TestNode(3, "three")
        node3 = TestNode(7, "seven")

        heap.insert(node1)
        heap.insert(node2)
        heap.insert(node3)

        # Multiple updates on same node
        heap.update(node1, 2)  # 5 -> 2
        heap.update(node1, 8)  # 2 -> 8
        heap.update(node1, 4)  # 8 -> 4

        # Order should be: 3, 4, 7
        self.assertEqual(heap.pop().key, 3)
        self.assertEqual(heap.pop().key, 4)
        self.assertEqual(heap.pop().key, 7)

    def test_same_node_multiple_operations_maxheap(self):
        """
        Test multiple operations on the same node in max heap.
        """
        heap = self.HeapClass(kind="max")
        node1 = TestNode(5, "five")
        node2 = TestNode(3, "three")
        node3 = TestNode(7, "seven")

        heap.insert(node1)
        heap.insert(node2)
        heap.insert(node3)

        # Multiple updates on same node
        heap.update(node1, 8)  # 5 -> 8
        heap.update(node1, 2)  # 8 -> 2
        heap.update(node1, 6)  # 2 -> 6

        # Order should be: 7, 6, 3
        self.assertEqual(heap.pop().key, 7)
        self.assertEqual(heap.pop().key, 6)
        self.assertEqual(heap.pop().key, 3)

    def test_mixed_operations_sequence(self):
        """
        Test a complex sequence of mixed operations.
        """
        for heap_kind in ["min", "max"]:
            heap = self.HeapClass(kind=heap_kind)
            nodes = {}

            # Build initial heap
            for i in [15, 10, 20, 8, 2, 12, 25]:
                nodes[i] = TestNode(i, str(i))
                heap.insert(nodes[i])

            # Perform series of operations
            heap.update(nodes[10], 30)  # Move 10 to 30
            heap.remove(nodes[20])  # Remove 20
            heap.insert(TestNode(5, "new5"))
            heap.update(nodes[8], 18)  # Move 8 to 18

            # Collect all remaining values
            result = []
            while True:
                try:
                    result.append(heap.pop().key)
                except (IndexError, KeyError, ValueError):
                    break

            # Verify correct ordering
            expected = [2, 5, 12, 15, 18, 25, 30]
            if heap_kind == "min":
                self.assertEqual(result, expected)
            else:  # max heap
                self.assertEqual(result, expected[::-1])


if __name__ == "__main__":
    unittest.main()
