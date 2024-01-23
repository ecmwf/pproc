import os
import sys
import json
import copy
import argparse

import numpy as np
from sklearn.cluster import DBSCAN
from io import BytesIO

import mir
import pyfdb
import eccodes
import pyinfero


class Utils:
    """
    A collection of generic TC utils
    """
    
    @staticmethod
    def ifs_steps(max_step):
        """
        Generates the correct sequence of steps for hres and ens
        :param max_step:
        :return:
        """
        
        return list(range(1, min(90, max_step + 1), 1)) + \
               list(range(90, min(144, max_step + 1), 3)) + \
               list(range(144, min(246, max_step + 1), 6))
    
    @staticmethod
    def pxl_2_lat_lon(pixel_indexes, fields_metadata):
        """
        From pixel indexes (+ fields info) to lat/lon coordinates
        """
        
        coords = []
        field_shape = fields_metadata["field_shape"]
        lat_min, lat_max = fields_metadata["latitude_range"]
        lon_min, lon_max = fields_metadata["longitude_range"]
        
        for tc in pixel_indexes:
            lat = lat_max - tc[0] / field_shape[0] * (lat_max - lat_min)
            lon = lon_min + tc[1] / field_shape[1] * (lon_max - lon_min)
            lon = lon - lon // 180 * 360
            coords.append((lat, lon))
        
        return coords
    
    @staticmethod
    def lat_lon_2_pxl(lat_lon, fields_metadata):
        """
        Function that maps lat/lon to pixel indexes
        (according to size and area range)
        """
        
        shape_y, shape_x = fields_metadata["field_shape"]
        lat_min, lat_max = fields_metadata["latitude_range"]
        lon_min, lon_max = fields_metadata["longitude_range"]
        
        pxl_idxs = []
        for lat, lon in lat_lon:
            pxl_x = int((lon - lon_min) / (lon_max - lon_min) * shape_x)
            pxl_x = min(max(0, pxl_x), shape_x - 1)
            
            pxl_y = int((lat_max - lat) / (lat_max - lat_min) * shape_y)
            pxl_y = min(max(0, pxl_y), shape_y - 1)
            
            pxl_idxs.append((pxl_y, pxl_x))
        
        return pxl_idxs


class JSONConfigurable:
    """
    A class configurable by JSON
    """
    
    # default config
    _default_config = {}
    _required_config_params = []
    
    def __init__(self, *args, **kwargs):
        
        # class own copy of config params
        self._config = copy.deepcopy(self._default_config)
        self._config.update(kwargs.get("config"))
        
        # check required params
        self.check_required_params()
        
        # Attribute first-level key/values
        for k, v in self._config.items():
            setattr(self, k, v)
    
    @classmethod
    def from_json(cls, json_file):
        
        if json_file is None:
            return cls({})
        
        try:
            with open(json_file, "r") as f:
                json_data = json.load(f)
        except FileNotFoundError:
            print(f"Json file {json_file} not found!")
            raise FileNotFoundError
        
        return cls(json_data)
    
    def check_required_params(self):
        """
        Check for needed parameters
        """
        
        for p in self._required_config_params:
            assert p in self._config.keys(), \
                f"Parameter {p} needed by {self.__class__}, " \
                f"but NOT found in configuration!"
    
    def save(self, file_path):
        """
        Save the configuration
        """
        with open(file_path, "w") as f:
            json.dump(self._config, f, indent=2, sort_keys=True)
            
    def summary(self):
        """
        Print a summary
        """
        print(json.dumps(self._config, indent=2, sort_keys=True))


class TCYCLConfig(JSONConfigurable):
    """
    Tcycl default configuration
    """
    
    _required_config_params = [
        "data_fetching",
        "annotations",
        "pre_processing",
        "inference",
        "cyclone_localizer"
    ]
    
    _default_config = {
      "data_fetching": {
        "retriever": "fdb",
        "variables": ["tcw", "100u", "100v", "sp", "sst", "lat", "lon"],
        "grid_resolution": [0.25, 0.25],
        "n_prev_steps": 3
      },
      "annotations": {
        "retriever": "fdb",
        "variables": ["10u", "10v", "sp"],
        "grid_resolution": [0.25, 0.25],
        "n_prev_steps": 1,
        "search_radius": 10
      },
      "pre_processing": {
        "lon_crop_pxl": 40,
        "lat_crop_pxl": 16,
        "norm_factor": {
          "tcw": 55,
          "100u": 25,
          "100v": 25,
          "sp": 100000,
          "sst": 300,
          "lat": 90,
          "lon": 180
        },
        "shift_factor": {
          "tcw": 55,
          "100u": 0,
          "100v": 0,
          "sp": 100000,
          "sst": 150,
          "lat": 0,
          "lon": 0
        }
      },
      "inference": {
        "model_path": "model.tflite",
        "model_type": "tflite",
        "model_output_shape": [1, 640, 1408, 1]
      },
      "cyclone_localizer": {
        "label_threshold": 0.6,
        "label_radius": 25
      }
    }
    
    def __init__(self, config):
        super().__init__(config=config)


class FieldData:
    """
    Single-field Data
    """
    
    required_keys = [
        "name",
        "date",
        "time",
        "step",
        "values",
        "resolution",
    ]
    
    def __init__(self, **kwargs):
        
        self._dict = kwargs
        
        for k in self.required_keys:
            try:
                setattr(self, k, kwargs[k])
            except KeyError:
                print(f"Key {k} is required but not found in {kwargs}!")
                raise KeyError
    
    @classmethod
    def from_dict(cls, dict_entry):
        """
        From a dictionary
        """
        return FieldData(**dict_entry)
    
    def to_dict(self):
        return self._dict


class FieldsData:
    """
    A list of FieldData with equal shape and resolution
    """
    
    # All fields MUST comply with the metadata
    metadata_fields = [
        "resolution",
        "field_shape",
        "latitude_range",
        "longitude_range",
    ]
    
    def __init__(self, fields, metadata):
        self.fields = fields
        self.metadata = metadata
        self._check_metadata()
    
    def _check_metadata(self):
        """
        Check for required metadata
        """
        for p in self.metadata_fields:
            assert p in self.metadata, f"Parameter {p} missing from metadata!"
    
    def _check_data(self):
        """
        Check data against corresponding metadata
        """
        
        for field in self.fields:
            assert field.values.shape == self.metadata["field_shape"]
            assert field.resolution == self.metadata["resolution"]
    
    def to_numpy(self):
        """
        Returns the concatenated fields as numpy
        """
        
        n = np.zeros((*self.metadata["field_shape"], len(self.fields)))
        for ff, field_data in enumerate(self.fields):
            n[:, :, ff] = field_data.values
        
        return n
    
    def update_metadata(self):
        """
        Attempt to update metadata after operating on values
        """
        
        # update field shape
        self.metadata["field_shape"] = self.fields[0].values.shape
        
        # update lat/lon range (if known)
        for f in self.fields:
            if f.name == "lat":
                self.metadata["latitude_range"] = np.min(f.values), np.max(f.values)
            if f.name == "lon":
                self.metadata["longitude_range"] = np.min(f.values), np.max(f.values)
        
        # check again all fields for consistency
        self._check_data()
    
    def get_fields(self, **kwargs):
        """
        Get fields with matching params
        """
        for k in kwargs.keys():
            assert k in FieldData.required_keys, f"Searching field by non-existent param {k}"
        
        fields = []
        for f in self.fields:
            if all([f.to_dict()[k] == v for k, v in kwargs.items()]):
                fields.append(f)
        
        return fields


class DataRetriever(JSONConfigurable):
    """
    Retrieve data
    """
    
    _required_config_params = [
        "variables",
        "grid_resolution",
        "n_prev_steps"
    ]
    
    def __init__(self, config):
        super().__init__(config=config)
    
    def get_data(self, request):
        """
        Get data
        """
        raise NotImplementedError


class FDBDataRetriever(DataRetriever):
    """
    Retrieve data from an FDB
    """
    
    # parameter name-id map
    p_map = {
        "tcw": "136",
        "10u": "165",
        "10v": "166",
        "100u": "228246",
        "100v": "228247",
        "sp": "134",
        "sst": "34"
    }
    
    # request template
    _req_template_base = {
        "date": None,
        "time": None,
        "class": "od",
        "expver": "0001",
        "levtype": "sfc",
        "step": None,
        "stream": None,
        "type": None,
        "domain": "g"
    }
    
    _req_template_hres = {
        "stream": "oper",
        "type": "fc"
    }
    
    _req_template_ens_pf = {
        "number": None,
        "stream": "enfo",
        "type": "pf"
    }
    
    _req_template_ens_cf = {
        "stream": "enfo",
        "type": "cf"
    }
    
    # required keys in GRIB message
    required_grib_keys = [
        "shortName",
        "Ni",
        "Nj",
        "dataDate",
        "dataTime",
        "stepRange"
    ]
    
    # required keys in user request
    _required_user_req_params = [
        "date",
        "time",
        "ens",
        "step"
    ]
    
    def __init__(self, config):
        super().__init__(config)
        
        # add necessary config params to request template
        self._req_template_base["param"] = [self.p_map.get(p)
                                            for p in config["variables"]
                                            if self.p_map.get(p)]
    
    @classmethod
    def _check_user_request(cls, request):
        """
        Check request params
        """
        for p in cls._required_user_req_params:
            assert p in request.keys(), \
                f"Parameter {p} needed by {cls}, " \
                f"but NOT found in request {request}!"
    
    def get_data_from_json_request(self, request_path):
        """
        Get data, from json request file name
        """
        with open(request_path, "r") as f:
            request = json.load(f)
        
        return self.get_data(request)
    
    def _compose_fdb_request(self, user_request):
        """
        Compose full FDB request
        """
        
        self._check_user_request(user_request)
        request = copy.deepcopy(self._req_template_base)
        
        assert isinstance(user_request["ens"], int), "ens field must be an int!"
        assert -1 <= user_request["ens"] <= 51, "Request ens must be in range [-1,50]. " \
                                                " Legend (-1:HRES, 0:cf, 1,50:pf)!"
        
        # date and time
        request["date"] = user_request["date"]
        request["time"] = user_request["time"]
        
        # fill-in required steps
        assert int(user_request["step"]) >= 3, f"Min step is 3." \
                                               f"Requested step {user_request['step']}!"
        
        steps = Utils.ifs_steps(int(user_request["step"]))
        
        # We might need to request few other steps before the selected steps
        # this is what the "self.n_prev_steps" parameter is for
        request["step"] = [str(s) for s in steps[-self.n_prev_steps:]]
        
        # -1 means HRES
        if user_request["ens"] == -1:
            request.update(self._req_template_hres)
        elif user_request["ens"] == 0:
            request.update(self._req_template_ens_cf)
        else:
            request.update(self._req_template_ens_pf)
            request["number"] = str(user_request["ens"])
        
        # No None fields must be left
        for k, v in request.items():
            assert v is not None, f"Request field {k} is None!"
        
        return request
    
    def get_data(self, user_request):
        """
        Get data, from json request
        """
        
        print("Retrieving data..")
        
        # generate full FDB request
        request = self._compose_fdb_request(user_request)
        print(f"Executing FDB request:\n{json.dumps(request, indent=2)}")
        
        # Expected N messages
        n_messages = len(request["param"]) * len(request["step"])
        _data = []
        
        fdb = pyfdb.FDB()
        reader = fdb.retrieve(request)
        
        grid_res_str = "/".join([str(v) for v in self.grid_resolution])
        stream = BytesIO()
        
        job = mir.Job(grid=str(grid_res_str))
        job.execute(reader, stream)
        
        stream.seek(0)
        ecc_reader = eccodes.StreamReader(stream)
        
        # # ====== With no MIR (for testing only) ======
        # fdb = pyfdb.FDB()
        # result = fdb.retrieve(request)
        # ecc_reader = eccodes.StreamReader(result)
        # # ============================================
        
        metadata = {}
        for mm, message in enumerate(range(n_messages)):
            
            try:
                message = next(ecc_reader)
            except StopIteration:
                print("ERROR: Failed to get the expected messages from FDB!")
                raise StopIteration
            
            _msg_data = self._grib_msg_to_field_data(message)
            _data.append(_msg_data)
            
            # append lat/lon at last
            if mm == n_messages - 1:
                lat_msg, lon_msg = self._grib_msg_to_latlon_data(message)
                _data.append(lat_msg)
                _data.append(lon_msg)
                
                metadata = {
                    "resolution": self.grid_resolution,
                    "field_shape": lat_msg.values.shape,
                    "latitude_range": (np.min(lat_msg.values), np.max(lat_msg.values)),
                    "longitude_range": (np.min(lon_msg.values), np.max(lon_msg.values))
                }
        
        # Compose FieldsData
        data = FieldsData(_data, metadata)
        
        return data
    
    def _grib_msg_to_field_data(self, grib_msg):
        """
        Read data from a GRIB message and
        return a FieldData object
        """
        
        value_shape = grib_msg.get("Nj"), grib_msg.get("Ni")
        val = grib_msg.get_array("values")
        field_values = np.asarray(val).reshape(value_shape)
        
        data_entry_dict = {
            "name": grib_msg.get("shortName"),
            "date": grib_msg.get("dataDate"),
            "time": grib_msg.get("dataTime"),
            "step": grib_msg.get("stepRange"),
            "values": field_values,
            "resolution": self.grid_resolution
        }
        
        return FieldData.from_dict(data_entry_dict)
    
    def _grib_msg_to_latlon_data(self, grib_msg):
        """
        Read lat/lon from a GRIB message and
        return 2 FieldData objects
        """
        
        _data = grib_msg.get_data_points()
        
        value_shape = grib_msg.get("Nj"), grib_msg.get("Ni")
        
        lat, lon = zip(*[(elem["lat"], elem["lon"]) for elem in _data])
        lat_values = np.asarray(lat).reshape(value_shape)
        lon_values = np.asarray(lon).reshape(value_shape)

        # lat dict
        lat_entry_dict = {
            "name": "lat",
            "date": grib_msg.get("dataDate"),
            "time": grib_msg.get("dataTime"),
            "step": grib_msg.get("stepRange"),
            "values": lat_values,
            "resolution": self.grid_resolution
        }
        
        # lon dict
        lon_entry_dict = {
            "name": "lon",
            "date": grib_msg.get("dataDate"),
            "time": grib_msg.get("dataTime"),
            "step": grib_msg.get("stepRange"),
            "values": lon_values,
            "resolution": self.grid_resolution
        }
        
        lat_field_data = FieldData.from_dict(lat_entry_dict)
        lon_field_data = FieldData.from_dict(lon_entry_dict)
        
        return lat_field_data, lon_field_data


# factory of data retrievers
data_retrievers = {
    "fdb": FDBDataRetriever
}


class PreProcessor(JSONConfigurable):
    """
    Preprocess data
    """
    
    _required_config_params = [
        "lon_crop_pxl",
        "lat_crop_pxl",
        "norm_factor",
        "shift_factor"
    ]
    
    def __init__(self, config):
        super().__init__(config=config)
    
    def run(self, input_data):
        print("Pre-processing data..")
        
        nj, ni = input_data.metadata["field_shape"]
        
        # field boundaries after cropping
        y0, y1 = self.lat_crop_pxl, ni - self.lat_crop_pxl
        x0, x1 = self.lon_crop_pxl, nj - self.lon_crop_pxl - 1
        
        # Crop fields
        fields = []
        for mm, field_data in enumerate(input_data.fields):
            processed_field = copy.deepcopy(field_data)
            processed_field.values = processed_field.values[x0:x1, y0:y1]
            fields.append(processed_field)
        
        # Update metadata for new shape and lat/lon range
        processed_metadata = copy.deepcopy(input_data.metadata)
        processed_data = FieldsData(fields, processed_metadata)
        processed_data.update_metadata()
        
        # Finally apply normalization/shift factors
        for mm, field_data in enumerate(processed_data.fields):
            param_name = field_data.name
            norm_factor = self.norm_factor[param_name]
            shift_factor = self.shift_factor[param_name]
            processed_data.fields[mm].values = (field_data.values - shift_factor) / norm_factor
        
        return processed_data


class Inference(JSONConfigurable):
    """
    Handle the execution of ML inference
    """
    
    _required_config_params = [
        "model_path",
        "model_type",
        "model_output_shape"
    ]
    
    def __init__(self, config):
        super().__init__(config=config)
    
    def run(self, input_data, save_to=None):
        """
        Runs Infero for inference
        """
        
        # add the outer (batch) dimension to tensor
        input_tensor = input_data.to_numpy()
        input_tensor = input_tensor[np.newaxis, ...]
        
        # inference
        infero = pyinfero.Infero(self.model_path, self.model_type)
        infero.initialise()
        output_tensor = infero.infer(input_tensor, self.model_output_shape)
        infero.finalise()
        
        # save when required
        if save_to is not None:
            np.save(save_to, output_tensor)
        
        return output_tensor


class CycloneLocalizer_DBSCAN(JSONConfigurable):
    """
    Uses DBSCAN algorithm
    """
    
    _required_config_params = [
        "label_threshold",
        "label_radius"
    ]
    
    def __init__(self, config):
        super().__init__(config=config)
    
    def find(self, prediction_data, metadata, save_to=None):
        """
        Apply kmeans to find cyclone centers
        """
        
        # remove 1-dimensions from prediction tensor
        prediction_data = np.squeeze(prediction_data)
        
        # points above threshold
        eye_mask = np.argwhere(prediction_data > self.label_threshold)
        if eye_mask.shape[0] <= 1:
            
            print("not enough points to attempt clustering..")
            clusters_lat_lon = []
            
            # save if required
            if save_to is not None:
                self._write_json(clusters_lat_lon, save_to)
                
            return clusters_lat_lon
        
        # Attempt clustering
        eye_weights = prediction_data[eye_mask[:, 0], eye_mask[:, 1]]
        dbscan = DBSCAN(self.label_radius).fit(eye_mask, sample_weight=eye_weights)
        
        clusters = []
        for label in set(dbscan.labels_):
            
            if label != -1:
                label_idxs = np.where(dbscan.labels_ == label)
                cluster_x = np.mean(eye_mask[label_idxs, 0])
                cluster_y = np.mean(eye_mask[label_idxs, 1])
                clusters.append((cluster_x, cluster_y))
        
        # transform to lat/lon
        clusters_lat_lon = Utils.pxl_2_lat_lon(clusters, metadata)
        
        # save when required
        if save_to is not None:
            self._write_json(clusters_lat_lon, save_to)
        
        return clusters_lat_lon
    
    @staticmethod
    def _write_json(clusters, save_path):
        with open(save_path, "w") as f:
            json.dump(clusters, f, indent=2, sort_keys=True)


class TCAnnotator(JSONConfigurable):
    """
    Produce annotations to TCYCL predictions
    (e.g. P_min, UV_max within a radius, etc..)
    """
    
    _required_config_params = [
        "retriever",
        "variables",
        "grid_resolution",
        "n_prev_steps",
        "search_radius"
    ]
    
    required_vars = [
        "10u",
        "10v",
        "sp"
    ]
    
    def __init__(self, config):
        super().__init__(config=config)
        
        # Variables required for annotation
        for var in self.required_vars:
            assert var in self.variables
    
    def run(self, predictions, data, save_to=None):
        """
        Find P_min and U_10 max within self.search_radius from TC's
        """
        
        tc_annotations = []
        predictions_pxl = Utils.lat_lon_2_pxl(predictions, data.metadata)
        
        values_sp = self._get_field(data, "sp")
        values_10u = self._get_field(data, "10u")
        values_10v = self._get_field(data, "10v")
        values_lat = self._get_field(data, "lat")
        values_lon = self._get_field(data, "lon")
        
        y_len, x_len = data.metadata["field_shape"]
        mesh_grid = np.meshgrid(np.arange(x_len), np.arange(y_len))
        mesh_zipped = np.stack((mesh_grid[0], mesh_grid[1]), axis=2)
        
        for pred in predictions_pxl:
            p = np.flip(np.asarray(pred)).reshape((1, 1, 2))  # pt coords
            diff = mesh_zipped - np.tile(p, (*data.metadata["field_shape"], 1))  # coord diff
            
            d2 = np.sum(diff ** 2, axis=2)  # square distance
            mask_idxs = np.where(d2 < self.search_radius ** 2)  # within-circle mask
            
            lats_in_mask = values_lat[mask_idxs]
            lons_in_mask = values_lon[mask_idxs]
            u10_in_mask = values_10u[mask_idxs]
            v10_in_mask = values_10v[mask_idxs]
            vabs_in_mask = (u10_in_mask ** 2 + v10_in_mask ** 2) ** 0.5
            sp_in_mask = values_sp[mask_idxs]
            
            # P_min
            p_min_idx = np.argmin(sp_in_mask)
            p_min = sp_in_mask[p_min_idx]
            p_min_lat = lats_in_mask[p_min_idx]
            p_min_lon = lons_in_mask[p_min_idx]
            print(f"p_min {p_min:.2f} (in lat:{p_min_lat:.2f}, lon:{p_min_lon:.2f})")
            
            # U10_max
            vabs_max_idx = np.argmax(vabs_in_mask)
            vabs_max = vabs_in_mask[vabs_max_idx]
            vabs_max_lat = lats_in_mask[vabs_max_idx]
            vabs_max_lon = lons_in_mask[vabs_max_idx]
            print(f"v_max {vabs_max:.2f} (in lat:{vabs_max_lat:.2f}, lon:{vabs_max_lon:.2f})")
            
            tc_annotations.append([p_min, p_min_lat, p_min_lon, vabs_max, vabs_max_lat, vabs_max_lon])
        
        # save when required
        if save_to is not None:
            with open(save_to, "w") as f:
                json.dump(tc_annotations, f, indent=2, sort_keys=True)
        
        return tc_annotations
    
    @staticmethod
    def _get_field(data, name):
        
        fields = data.get_fields(name=name)
        assert len(fields) == 1, "only 1 field expected!"
        return np.asarray(fields[0].values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_request", help=f"Path to JSON data request")
    
    parser.add_argument("--config", help=f"JSON of tcycl user-configuration")
    parser.add_argument("--save_ml_output", help=f"Path to numpy ML prediction file")
    parser.add_argument("--save_prediction", help=f"Path to JSON prediction file")
    parser.add_argument("--annotate_prediction", help=f"Path to JSON annotation file")
    
    args = parser.parse_args()
    
    # Tcycl configuration
    config = TCYCLConfig.from_json(args.config)
    config.summary()
    
    # Get input data
    retriever_type = data_retrievers[config.data_fetching["retriever"]]
    retriever = retriever_type(config.data_fetching)
    retrieved_data = retriever.get_data_from_json_request(args.json_request)
    
    # Pre-process input data
    pre_processor = PreProcessor(config.pre_processing)
    pre_proc_data = pre_processor.run(retrieved_data)
    
    # Run Inference
    inference_runner = Inference(config.inference)
    output_data = inference_runner.run(pre_proc_data, save_to=args.save_ml_output)
    
    # Run Clustering
    localizer = CycloneLocalizer_DBSCAN(config.cyclone_localizer)
    prediction = localizer.find(output_data, pre_proc_data.metadata, save_to=args.save_prediction)
    
    # Annotate prediction
    if args.annotate_prediction:
        annotation_data_retriever = data_retrievers[config.data_fetching["retriever"]](config.annotations)
        annotation_data = annotation_data_retriever.get_data_from_json_request(args.json_request)
        annotator = TCAnnotator(config.annotations)
        annotations = annotator.run(prediction, annotation_data, save_to=args.annotate_prediction)
    
    print(f"All done.")


if __name__ == "__main__":
    sys.exit(main())
