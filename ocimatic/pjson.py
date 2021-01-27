import json


def load(file_path):
    return PJSONFile(file_path).load()


class PJSONFile:
    def __init__(self, file_path):
        self._file_path = file_path

    def _load_json(self):
        if not self._file_path.exists():
            return {}
        with self._file_path.open('r') as json_file:
            return json.load(json_file)

    def _dump_json(self, obj):
        with self._file_path.open('w') as json_file:
            return json.dump(obj, json_file, indent=4)

    def get_path(self, path):
        val = self._load_json()
        for key in path:
            val = val[key]
        return val

    def set_path(self, path, val):
        if path:
            root = self._load_json()
            obj = root
            for key in path[:-1]:
                obj = obj[key]
            obj[path[-1]] = val
        else:
            root = val
        self._dump_json(root)

    def load(self):
        return self.to_pjson(self._load_json(), [])

    def to_pjson(self, val, path):
        if isinstance(val, list):
            return PJSONArray(self, path)
        if isinstance(val, dict):
            return PJSONMap(self, path)
        return val


class PJSONBase:
    def __init__(self, json_file, path):
        self._json_file = json_file
        self._path = path

    def _get_self_path(self):
        return self._json_file.get_path(self._path)

    def _set_self_path(self, val):
        self._json_file.set_path(self._path, val)

    def __getitem__(self, key):
        return self._json_file.to_pjson(self._get_self_path()[key], self._path + [key])

    def __setitem__(self, key, val):
        obj = self._get_self_path()
        obj[key] = val
        self._set_self_path(obj)
        return self[key]

    def __contains__(self, val):
        return val in self._get_self_path()


class PJSONMap(PJSONBase):
    def __iter__(self):
        obj = self._get_self_path()
        for key, val in obj:
            yield (key, val)

    def setdefault(self, key, default=None):
        obj = self._get_self_path()
        if key not in obj:
            self[key] = default
        return self[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class PJSONArray(PJSONBase):
    def append(self, val):
        arr = self._get_self_path()
        arr.append(val)
        self._set_self_path(arr)

    def __len__(self):
        return len(self._get_self_path())

    def __iter__(self):
        arr = self._get_self_path()
        for val in arr:
            yield val
