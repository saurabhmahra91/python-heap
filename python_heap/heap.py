from collections.abc import Hashable
from typing import Literal, Protocol


class PQNode(Hashable, Protocol):
    """
    Protocol representing a node that can be stored in a Heap.

    Attributes:
        key (int | float): The value used for ordering in the heap.

    Notes:
        - PQNodes must be unique and hashable to allow O(1) access and updates in the heap.
    """

    key: int | float


class Heap:
    """
    A binary heap implementation supporting both min-heap and max-heap behavior.

    This heap allows:
        - Insertions in O(log n)
        - Updates in O(log n)
        - Pops (removing top element) in O(log n)
        - Removal of arbitrary elements in O(log n)

    Attributes:
        kind (Literal["min", "max"]): Determines whether the heap is min-heap or max-heap.
        _heap_array (list[PQNode]): Internal array storing heap elements.
        _index_dict (dict[PQNode, int]): Maps nodes to their indices for O(1) access and updates.

    Notes:
        - PQNodes must be unique and hashable.
    """

    def __init__(self, kind: Literal["min", "max"] = "min") -> None:
        """
        Initialize the heap.

        Args:
            kind (Literal["min", "max"]): Type of heap. Defaults to "min".
        """
        assert kind in {"min", "max"}

        self.kind = kind
        self._heap_array: list[PQNode] = []
        self._index_dict: dict[PQNode, int] = dict()

    @staticmethod
    def parent(ind: int) -> int:
        """
        Return the index of the parent node of a given index.
        """
        if ind == 0:
            return 0
        return (ind - 1) // 2

    def __len__(self):
        """
        Return the number of elements in the heap.
        """
        return len(self._heap_array)

    @staticmethod
    def lc(ind: int) -> int:
        """
        Return the index of the left child of a given node index.
        """
        return 2 * ind + 1

    @staticmethod
    def rc(ind: int) -> int:
        """
        Return the index of the right child of a given node index.
        """
        return 2 * ind + 2

    def _swap(self, a: int, b: int):
        """
        Swap two nodes in the heap and update their indices.
        """
        self._index_dict[self._heap_array[a]], self._index_dict[self._heap_array[b]] = (
            self._index_dict[self._heap_array[b]],
            self._index_dict[self._heap_array[a]],
        )
        self._heap_array[a], self._heap_array[b] = self._heap_array[b], self._heap_array[a]

    def _sift_up(self, ind: int):
        """
        Restore heap property by moving a node up.
        """
        if ind == 0:
            return

        swap_cond = self._heap_array[ind].key < self._heap_array[Heap.parent(ind)].key
        if self.kind == "max":
            swap_cond = not swap_cond

        if swap_cond:
            self._swap(a=ind, b=Heap.parent(ind))
            self._sift_up(Heap.parent(ind))

    def _sift_down(self, ind: int) -> None:
        """
        Restore heap property by moving a node down.
        """
        has_lc = Heap.lc(ind) < len(self)
        has_rc = Heap.rc(ind) < len(self)

        if not has_lc:
            return

        elif has_lc and not has_rc:
            swap_cond = self._heap_array[Heap.lc(ind)].key < self._heap_array[ind].key

            if self.kind == "max":
                swap_cond = not swap_cond

            if swap_cond:
                self._swap(ind, Heap.lc(ind))
                self._sift_down(Heap.lc(ind))

        elif has_lc and has_rc:
            swap_c = Heap.lc if self._heap_array[Heap.lc(ind)].key < self._heap_array[Heap.rc(ind)].key else Heap.rc

            if self.kind == "max":
                swap_c = Heap.lc if self._heap_array[Heap.lc(ind)].key > self._heap_array[Heap.rc(ind)].key else Heap.rc
                should_swap = self._heap_array[swap_c(ind)].key > self._heap_array[ind].key

            elif self.kind == "min":
                swap_c = Heap.rc if self._heap_array[Heap.lc(ind)].key > self._heap_array[Heap.rc(ind)].key else Heap.lc
                should_swap = self._heap_array[swap_c(ind)].key < self._heap_array[ind].key
            else:
                raise ValueError(f"Unknown Heap kind `{self.kind}`")

            if should_swap:
                self._swap(swap_c(ind), ind)
                self._sift_down(swap_c(ind))

    def insert(self, n: PQNode) -> None:
        """
        Insert a new node into the heap.
        """
        if not isinstance(n, Hashable):
            raise TypeError(f"PQNode must be hashable, got {type(n)}")

        import warnings

        if n in self._index_dict:
            warnings.warn("PQNode is already in the heap; duplicates are discouraged.", UserWarning, stacklevel=2)

        self._heap_array.append(n)
        self._index_dict[n] = len(self._heap_array) - 1
        self._sift_up(len(self._heap_array) - 1)

    def update(self, n: PQNode, key: int | float):
        """
        Update the key of an existing node and restore heap property.

        Args:
            n (PQNode): Node to update.
            key (int | float): New key value.
        """
        old_key = self._heap_array[self._index_dict[n]].key
        self._heap_array[self._index_dict[n]].key = key

        if key > old_key:
            if self.kind == "min":
                self._sift_down(self._index_dict[n])
            else:
                self._sift_up(self._index_dict[n])

        if key < old_key:
            if self.kind == "min":
                self._sift_up(self._index_dict[n])
            else:
                self._sift_down(self._index_dict[n])

    def pop(self) -> PQNode:
        """
        Remove and return the top node (min or max depending on heap kind).
        """
        from copy import deepcopy

        self._swap(a=0, b=len(self) - 1)
        response = deepcopy(self._heap_array[-1])

        del self._index_dict[self._heap_array[-1]]
        del self._heap_array[-1]

        self._sift_down(ind=0)
        return response

    def remove(self, n: PQNode) -> None:
        """
        Remove a specific node from the heap.
        """
        key_backup = self._heap_array[self._index_dict[n]].key

        self.update(n, key=float("-inf") if self.kind == "min" else float("inf"))
        self.pop()

        n.key = key_backup
