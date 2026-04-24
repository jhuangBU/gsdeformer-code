import asyncio
from typing import TypeVar, Generic, Callable, List, Coroutine

T = TypeVar("T")


class EventEmitter(Generic[T]):

    def __init__(self):
        self._callbacks = []

    def register(self, function: Callable[[T], None]) -> Callable[[], None]:
        self._callbacks.append(function)
        return lambda: self._callbacks.remove(function)

    def emit(self, val: T):
        for f in self._callbacks:
            f(val)
