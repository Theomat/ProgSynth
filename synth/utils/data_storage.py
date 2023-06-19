import bz2
import pickle
from typing import Any, Callable, Optional
import pickletools

import _pickle as cPickle  # type: ignore


def load_object(
    path: str, unpickler: Optional[Callable[[bz2.BZ2File], pickle.Unpickler]] = None
) -> Any:
    """
    Load an arbitrary object from the specified file.
    """
    with bz2.BZ2File(path, "rb") as fd:
        if unpickler is None:
            return cPickle.load(fd)
        else:
            return unpickler(fd).load()


def save_object(path: str, obj: Any, optimize: bool = True) -> None:
    """
    Save an arbitrary object to the specified path.
    """
    with bz2.BZ2File(path, "w") as fd:
        content = pickle.dumps(obj)
        if optimize:
            content = pickletools.optimize(content)
        fd.write(content)
