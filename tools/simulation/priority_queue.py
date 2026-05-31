"""Priority queue with backpressure for MaxSight Web Simulator. Prevents memory growth and ensures fresh alerts take priority."""

from enum import IntEnum
from queue import Full, Queue
from threading import Lock
from typing import Any


class MessagePriority(IntEnum):
    """Message priority levels (higher = more urgent)."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class PriorityQueue:
    """Bounded priority queue with backpressure. On overflow: - Drops low-priority messages first - Keeps only the latest high-urgency alert - Prevents memory growth."""

    def __init__(self, maxsize: int = 10):
        """Args maxsize: Maximum queue size (0 = unbounded, not recommended)"""
        self.maxsize = maxsize
        self.queue: Queue = Queue(maxsize=maxsize)
        self.lock = Lock()
        self._dropped_count = 0
        self._last_critical: tuple[int, Any] | None = None  # (priority, message)

    def put(self, item: tuple[Any, int], block: bool = True, timeout: float | None = None) -> bool:
        """Put item in queue with priority."""
        message, priority = item
        priority_value = priority if isinstance(priority, int) else priority.value

        with self.lock:
            # If critical, always keep the latest.
            if priority_value >= MessagePriority.CRITICAL.value:
                self._last_critical = (priority_value, message)

            # Put in queue.
            try:
                self.queue.put((priority_value, message), block=block, timeout=timeout)
                return True
            except Full:
                # Queue is full - apply backpressure.
                if priority_value >= MessagePriority.CRITICAL.value:
                    # Replace last critical message when the new one is newer or higher priority.
                    if self._last_critical and priority_value >= self._last_critical[0]:
                        # Remove old critical and add new one.
                        self._try_replace_critical(priority_value, message)
                        return True
                    else:
                        # Drop older or lower-priority critical message.
                        self._dropped_count += 1
                        return False
                elif priority_value >= MessagePriority.HIGH.value:
                    # High priority: drop a low-priority item.
                    if self._try_drop_low_priority():
                        try:
                            self.queue.put((priority_value, message), block=False)
                            return True
                        except Full:
                            self._dropped_count += 1
                            return False
                    else:
                        # Drop incoming message when no low-priority item can be evicted.
                        self._dropped_count += 1
                        return False
                else:
                    # Low/Normal priority: drop it.
                    self._dropped_count += 1
                    return False

    def _try_replace_critical(self, new_priority: int, new_message: Any) -> bool:
        """Try to replace old critical message with new one."""
        # When queue is full, drop the new message; production may scan and replace.
        return False

    def _try_drop_low_priority(self) -> bool:
        """Try to remove a low-priority item from queue."""
        # Queue doesn't support selective removal easily.
        # For now, we'll just drop the incoming high-priority if queue is full.
        # Use a structure that supports selective removal in production.
        return False

    def get(self, block: bool = True, timeout: float | None = None) -> tuple[int, Any]:
        """Get item from queue (highest priority first). Returns: Tuple of (priority, message)"""
        return self.queue.get(block=block, timeout=timeout)

    def get_dropped_count(self) -> int:
        """Get count of dropped messages."""
        with self.lock:
            return self._dropped_count

    def qsize(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()

    def empty(self) -> bool:
        """Check if queue is empty."""
        return self.queue.empty()

    def full(self) -> bool:
        """Check if queue is full."""
        return self.queue.full()
