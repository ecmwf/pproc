from earthkit.data.readers.grib import metadata
from earthkit.data.readers.grib import memory
from earthkit.data.readers.grib import codes
from earthkit.data import FieldList
from earthkit.data.sources import numpy_list


class GribMetadata(metadata.GribMetadata):
    def __init__(self, handle, clear_data: bool = False):
        if clear_data:
            # Clear data values, keeping only metadata
            handle = handle.clone()
            handle.set_array("values", handle.get_array("values").shape)
        super().__init__(handle)

    def __getstate__(self) -> dict:
        ret = self.__dict__.copy()
        ret["_handle"] = self._handle.get_buffer()
        return ret

    def __setstate__(self, state: dict):
        state["_handle"] = codes.GribCodesHandle(
            memory.GribMessageMemoryReader(state["_handle"])._next_handle(), None, None
        )
        self.__dict__.update(state)

    def _hide_internal_keys(self):
        return self


class ArrayFieldList(numpy_list.NumpyFieldList):
    def __init__(self, array, metadata):
        metadata = [GribMetadata(x._handle) for x in metadata]
        super().__init__(array, metadata)

    def __getstate__(self) -> dict:
        return {
            "array": self.values,
            "metadata": self.metadata(),
        }

    def __setstate__(self, state: dict):
        new_fieldlist = FieldList.from_numpy(state["array"], state["metadata"])
        self.__dict__.update(new_fieldlist.__dict__)
        del new_fieldlist
