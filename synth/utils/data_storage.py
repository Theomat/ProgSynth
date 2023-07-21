import bz2
import pickle
from typing import Any, Callable, Optional
import pickletools


def load_object(
    path: str, unpickler: Optional[Callable[[bz2.BZ2File], pickle.Unpickler]] = None
) -> Any:
    """
    Load an arbitrary object from the specified file.
    """
    with bz2.BZ2File(path, "rb") as fd:
        if unpickler is None:
            return pickle.load(fd)
        else:
            return unpickler(fd).load()


def save_object(
    path: str, obj: Any, optimize: bool = True, compress_level: int = 9
) -> None:
    """
    Save an arbitrary object to the specified path.
    Compression level must be in 1-9 where 9 is the highest level.
    """
    with bz2.BZ2File(path, "w", compresslevel=compress_level) as fd:
        content = pickle.dumps(obj)
        if optimize:
            content = pickletools.optimize(content)
        fd.write(content)
