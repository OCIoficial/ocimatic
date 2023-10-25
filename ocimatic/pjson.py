import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def load(file_path: Path) -> Any:
    return PJSONFile(file_path).load()


class PJSONFile:
    def __init__(self, file_path: Path) -> None:
        self._file_path = file_path

    def _load_json(self) -> Any:
        if not self._file_path.exists():
            return {}
        with self._file_path.open("r") as json_file:
            return json.load(json_file)

    def _dump_json(self, obj: Any) -> Any:
        with self._file_path.open("w") as json_file:
            return json.dump(obj, json_file, indent=4)

    def get_path(self, path: Any) -> Any:
        val = self._load_json()
        for key in path:
            val = val[key]
        return val

    def set_path(self, path: list[str], val: Any) -> Any:
        if path:
            root = self._load_json()
            obj = root
            for key in path[:-1]:
                obj = obj[key]
            obj[path[-1]] = val
        else:
            root = val
        self._dump_json(root)

    def load(self) -> Any:
        return self.to_pjson(self._load_json(), [])

    def to_pjson(self, val: Any, path: list[str]) -> Any:
        if isinstance(val, list):
            return PJSONArray(self, path)
        if isinstance(val, dict):
            return PJSONMap(self, path)
        return val


class PJSONBase:
    def __init__(self, json_file: PJSONFile, path: list[str]) -> None:
        self._json_file = json_file
        self._path = path

    def _get_self_path(self) -> Any:
        return self._json_file.get_path(self._path)

    def _set_self_path(self, val: Any) -> Any:
        self._json_file.set_path(self._path, val)

    def __getitem__(self, key: str) -> Any:
        return self._json_file.to_pjson(self._get_self_path()[key], [*self._path, key])

    def __setitem__(self, key: str, val: Any) -> Any:
        obj = self._get_self_path()
        obj[key] = val
        self._set_self_path(obj)
        return self[key]

    def __contains__(self, val: Any) -> bool:
        return val in self._get_self_path()


class PJSONMap(PJSONBase):
    def __iter__(self) -> Iterable[tuple[str, Any]]:
        yield from self._get_self_path()

    def setdefault(self, key: str, default: Any = None) -> Any:
        obj = self._get_self_path()
        if key not in obj:
            self[key] = default
        return self[key]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class PJSONArray(PJSONBase):
    def append(self, val: Any) -> None:
        arr = self._get_self_path()
        arr.append(val)
        self._set_self_path(arr)

    def __len__(self) -> int:
        return len(self._get_self_path())

    def __iter__(self) -> Iterable[Any]:
        yield from self._get_self_path()
