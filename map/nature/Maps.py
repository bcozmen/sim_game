import numpy as np


_to_dtype = {
    0: np.float32,
    1: np.bool_,
    2: np.int32,
    "float32": np.float32,
    "float": np.float32,
    "int32": np.int32,
    "int": np.int32,
    "bool": np.bool_,
}


class Map:
    def __init__(self, size):
        self.size = size
        self.info = {}
        self.data = {
            np.float32 : np.zeros(size + (0,), dtype=np.float32),
            np.int32 : np.zeros(size + (0,), dtype=np.int32),
            np.bool_ : np.zeros(size + (0,), dtype=np.bool_),
        }

    def add_map(self, name, dtype):
        dtype = _to_dtype[dtype]
        self.data[dtype] = np.concatenate((self.data[dtype], np.zeros(self.size + (1,), dtype=dtype)), axis=-1)
        self.info[name] = (dtype, self.data[dtype].shape[-1] - 1)
    def __getitem__(self, key):
        dtype, idx = self.info[key]
        return self.data[dtype][..., idx]
    
    def _validate_value(self, key, value):
        dtype, idx = self.info[key]
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        if value.shape != self.size:
            raise ValueError(f"Value for key '{key}' must have shape {self.size}, but got {value.shape}")
        if value.dtype != dtype:
            raise ValueError(f"Value for key '{key}' must have dtype {dtype}, but got {value.dtype}")
        return value
    def __setitem__(self, key, value):
        value = self._validate_value(key, value)
        dtype, idx = self.info[key]
        self.data[dtype][..., idx] = value

    
shape = (1024, 1024)
maps = Map(shape)
float_map_1 = np.random.rand(*shape).astype(np.float32)
int_map_1 = np.random.randint(0, 100, size=shape, dtype=np.int32)
bool_map_1 = np.random.choice([False, True], size=shape)
maps.add_map("elevation", "float32")
maps.add_map("vegetation", "int32")
maps.add_map("water", "bool")
maps["elevation"] = float_map_1
maps["vegetation"] = int_map_1
maps["water"] = bool_map_1