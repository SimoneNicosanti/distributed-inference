from collections.abc import Callable
from typing import TypeVar

F = TypeVar("F", bound=Callable[..., object])


def raises(*exceptions: type[Exception]) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        documented = "\n".join(f"    {exc.__name__}" for exc in exceptions)

        func.__doc__ = (func.__doc__ or "") + f"\n\nRaises:\n{documented}"
        return func

    return decorator
