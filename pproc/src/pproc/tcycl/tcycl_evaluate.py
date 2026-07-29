# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import sys
import math
import argparse

import numpy as np
import pandas as pd


class TropicalCyclone:
    """
    Simply a TC with name and lat/lon
    """

    allowed_params = [
        "p",
        "wind",
        "step"
    ]
    
    def __init__(self, name, lat, lon, **kwargs):
    
        self.name = name
        self.lat = lat
        self.lon = lon
        self.params = kwargs
        
        # check other TC params
        self._check_params(kwargs)
        
    def _check_params(self, params):
        
        for p in params:
            assert p in self.allowed_params, f"Parameter {p} not allowed!"
    
    def distance_from(self, tc_other):
        return self._haversine_dist(self.lat,
                                    self.lon,
                                    tc_other.lat,
                                    tc_other.lon)

    @staticmethod
    def _haversine_dist(lat1, lon1, lat2, lon2):
        """
        Haversine distance
        """
        p = math.pi / 180
        
        a = 0.5 - \
            math.cos((lat2 - lat1) * p) / 2 + \
            math.cos(lat1 * p) * math.cos(lat2 * p) * \
            (1 - math.cos((lon2 - lon1) * p)) / 2

        # 2 * R * asin...
        distance = 12742 * math.asin(math.sqrt(a))
        
        return distance
    
    def __str__(self):
        """
        
        """
        
        min_req_str = f"{self.name:15} lat: {self.lat:7.2f} lon: {self.lon:7.2f}, "
        params_str = ", ".join([f'{k}: {w:7.2f}' for k, w in self.params.items()])
        
        return min_req_str + params_str


class TC_Forecast:
    """
    Generic TC forecast data - mainly a
    dictionary of tropical cyclones
    grouped according to forecast step
    """
    
    def __init__(self, *args, **kwargs):
 
        # Structure of self.data
        # {
        #   <step1>: [
        #   <TropicalCyclone1>,
        #   <TropicalCyclone2>,
        #   ...
        #   ],
        #   <step2>: [
        #   <TropicalCycloneN>,
        #   ...
        #   ],
        #   ...
        # }
        self.data = self._parse()
        
    def _parse(self):
        raise NotImplementedError
    
    def __getitem__(self, item):
        return self.data.get(item, [])
    
    @property
    def steps(self):
        return sorted(list(self.data.keys()))

    def print(self):
        """
        Print TC's
        """
        for s, tc_s in self.data.items():
            for _tc in tc_s:
                print(_tc)

                
class TC_Observation(TC_Forecast):
    """
    Similar to TC forecast but with no
    associated "step"
    """

    # Structure of self.data
    # [
    #   <TropicalCyclone1>,
    #   <TropicalCyclone2>,
    #   ...
    # ]
    
    def __getitem__(self, item):
        """
        Always return the data, regardless of the "step",
        as there is no concept of "step" for tc observation
        """
        return self.data
    
    @property
    def steps(self):
        return []
    
    def print(self):
        """
        Print TC's
        """
        for tc_s in self.data:
            for _tc in tc_s:
                print(_tc)
                

class TRCKR_Forecast(TC_Forecast):
    """
    Reference TC data from CSV
    """
    
    required_fields = [
        "name",
        "step",
        "lat_p",
        "lon_p"
    ]
    
    def __init__(self,
                 reference_tc_path,
                 min_wind_speed=None,  # 34 knots = 17.47 m/s
                 search_steps=None):
        
        # Reference TC predictions read from CSV
        self.df = pd.read_csv(reference_tc_path)
        
        if min_wind_speed is not None:
            self._filter_out_low_wind_tc(min_wind_speed)
        
        # Check data
        self._check_csv_data()
        
        if search_steps is not None:
            self.search_steps = search_steps
        else:
            self.search_steps = sorted(set(self.df.step))
        
        super().__init__()
        
    def _filter_out_low_wind_tc(self, min_wind_speed):
        self.df = self.df.loc[self.df['wind'] >= min_wind_speed]
        
    def _check_csv_data(self):
        """
        Check that required fields are in CSV
        """
        
        for req in self.required_fields:
            assert req in self.df, f"Error, field {req} not found in CSV!"
        
    def _parse(self):
        
        # Collect TC tracker prediction data
        tc_predictions = {}
        for step in self.search_steps:
            tc_predictions[step] = [TropicalCyclone(name, lat, lon, wind=w, p=p, step=step)
                                    for name, lat, lon, w, p in self._tc_step_prediction(step)]
            
        return tc_predictions

    def _tc_step_prediction(self, step):
        """
        Single step-data
        """
        
        tc_id = self.df.loc[self.df['step'] == step].name
        lats = self.df.loc[self.df['step'] == step].lat_p
        lons = self.df.loc[self.df['step'] == step].lon_p
        wind = self.df.loc[self.df['step'] == step].wind
        pres = self.df.loc[self.df['step'] == step].pres
    
        step_lats_lons = list(zip(tc_id, lats, lons, wind, pres))
    
        return step_lats_lons


class TCYCL_Forecast(TRCKR_Forecast):
    """
    Predicted TC's from TCYCL tool
    """
    
    required_fields = [
        "name",
        "step",
        "lat",
        "lon"
    ]
    
    additional_fields = [
        "P_min",
        "P_min_lat",
        "P_min_lon",
        "V10_max",
        "V10_max_lat",
        "V10_max_lon"
    ]
    
    def __init__(self,
                 reference_tc_path,
                 min_wind_speed=None,
                 search_steps=None):
                
        super().__init__(reference_tc_path,
                         min_wind_speed=min_wind_speed,
                         search_steps=search_steps)
        
    def _filter_out_low_wind_tc(self, min_wind_speed):
        self.df = self.df.loc[self.df['V10_max'] >= min_wind_speed]

    def _parse(self):
        # Collect TC tracker prediction data
        tc_predictions = {}
        for step in self.search_steps:
            tc_predictions[step] = [TropicalCyclone(name, lat, lon, wind=w, p=p, step=step)
                                    for name, lat, lon, w, p in self._tc_step_prediction(step)]
    
        return tc_predictions

    def _tc_step_prediction(self, step):
        """
        Single step-data
        """
    
        tc_id = self.df.loc[self.df['step'] == step].name
        lats = self.df.loc[self.df['step'] == step].lat
        lons = self.df.loc[self.df['step'] == step].lon
        wind = self.df.loc[self.df['step'] == step].V10_max
        pres = self.df.loc[self.df['step'] == step].P_min
    
        step_lats_lons = list(zip(tc_id, lats, lons, wind, pres))
    
        return step_lats_lons


class IBTKS_Data(TC_Observation):
    
    required_fields = [
        "ISO_TIME",
        "SID",
        "NUMBER",
        "BASIN",
        "SUBBASIN",
        "NAME",
        "NATURE",
        "LAT",
        "LON",
        "wind",
        "pres",
    ]
    
    additional_fields = []

    def __init__(self,
                 reference_tc_path,
                 min_wind_speed=None):
        
        self.df = pd.read_csv(reference_tc_path)

        if min_wind_speed is not None:
            self._filter_out_low_wind_tc(min_wind_speed)

        # Check data
        self._check_csv_data()
        
        super().__init__()

    def _filter_out_low_wind_tc(self, min_wind_speed):
        self.df = self.df.loc[self.df['wind'] >= min_wind_speed]

    def _check_csv_data(self):
        """
        Check that required fields are in CSV
        """
    
        for req in self.required_fields:
            assert req in self.df, f"Error, field {req} not found in CSV!"

    def _parse(self):
        
        # Collect TC tracker prediction data
        tc_data = [TropicalCyclone(name, lat, lon, wind=w, p=p)
                   for name, lat, lon, w, p in self._tc_step_prediction()]
    
        return tc_data

    def _tc_step_prediction(self):
        """
        Single step-data
        """
    
        # here we take ALL the values in the df (step not usable as
        # ibtracks are historical data and not FC). Also, it is assumed
        # that ibtracks summary contains ONLY the data we want to compare
        tc_id = self.df.SID
        lats = self.df.LAT
        lons = self.df.LON
        wind = self.df.wind
        pres = self.df.pres
    
        step_lats_lons = list(zip(tc_id, lats, lons, wind, pres))
   
        return step_lats_lons


class Scorer:
    """
    Gives a score to TC prediction(s)
    """
    
    def __init__(self, *args, **kwargs):
        self._score = None
    
    def run(self):
        raise NotImplementedError
    
    def score(self):
        return self._score
    
    def summary(self):
        raise NotImplementedError


class ConfusionMatrix(Scorer):
    """
    Calculates confusion matrix from reference and TCYCL data
    by approximating a match between real and predicted TC
    """
    
    # ====== Confusion matrix ======== #
    #            ___________________   #
    #   _________| P_pred | N_pred |   #
    #   | P_real |   tp   |   fn   |   #
    #   | N_real |   fp   |   tn   |   #
    #   ----------------------------   #
    #                                  #
    # ================================ #
    
    def __init__(self,
                 ref_data,
                 tcycl_data,
                 threshold,
                 compare_at_steps=None):
        
        self.ref_data = ref_data
        self.tcycl_data = tcycl_data
        self.threshold = threshold
        
        if compare_at_steps is None:
            self.compare_at_steps = self.tcycl_data.steps
        else:
            self.compare_at_steps = compare_at_steps
        
        super().__init__()
        
    def summary(self):
        """
        Print a summary
        """
        
        sums = {k: len(v) for k, v in self._score.items()}
        avgs = {k: len(v)/len(self.compare_at_steps) for k, v in self._score.items()}

        print(f"\n")
        print(f"#################################")
        print(f"#####        Summary        #####")
        print(f"#################################")
        print(f"\n")
        self._print_score()

        print(f"")
        print(f" ================ SUM ================")
        if sums:
            self._print_matrix(sums)
        else:
            print("No predictions found.")
        print(f" ================ AVG ================")
        if avgs:
            self._print_matrix(avgs)
        else:
            print("No predictions found.")
        print(f"")

    @staticmethod
    def _print_matrix(vals):
        """
        Print a confusion matrix
        """
        
        tp = f"{vals['true-positives']:^12.2f}"
        fp = f"{vals['false-positives']:^12.2f}"
        fn = f"{vals['false-negatives']:^12.2f}"
        na = f"    N/A     "
        
        print(f"")
        print(f"           ---------------------------  ")
        print(f"           |   P_pred   |   N_pred   | ")
        print(f"  -----------------------------------| ")
        print(f"  | P_real |{tp}|{fn}| ")
        print(f"  |----------------------------------| ")
        print(f"  | N_real |{fp}|{na}| ")
        print(f"  ------------------------------------ ")
        print(f"")
        
    def _print_score(self):
        
        if self._score is None:
            print(f"Score cannot be printed. Invoke Run first.")
            raise ValueError
        
        # print the score
        for k, v in self._score.items():
            print(f"{k} [{len(v)}]:")
            print(*[str(t) for t in v], sep='\n')
        
    def run(self):
        
        print(f"\n")
        print(f"#################################")
        print(f"##### Starting evaluation.. #####")
        print(f"#################################")
        print(f"\n")
    
        # Calculate confusion matrix for all the steps
        self._score = {}
        for step in self.compare_at_steps:
        
            s_trckr = self.ref_data[step]
            s_tcycl = self.tcycl_data[step]

            print(f"#### Step {step}:")
            print(f"--> TRACK-cyclones ({len(s_trckr)}):", *[str(t) for t in s_trckr], sep='\n')
            print(f"--> TCYCL-cyclones ({len(s_tcycl)}):", *[str(t) for t in s_tcycl], sep='\n')
        
            step_scores = self._run_step(s_trckr, s_tcycl, self.threshold)
            
            # Extend tp, fp, fn
            for k, v in step_scores.items():
                self._score.setdefault(k, []).extend(v)

    @staticmethod
    def _run_step(ref_step_data, tcycl_data_step, threshold):
        
        matched = []
        if len(ref_step_data):
            print(f"\n# Matching TC's..")
            
        for trckr_tc in ref_step_data:
            
            print(f"Processing TRACKR cyclone:\n{trckr_tc}")
            
            min_dist = 1e10
            closest_tcycl = None
            for tcycl_tc in tcycl_data_step:
                dist = trckr_tc.distance_from(tcycl_tc)
                
                print(f"TCYCL cyclone: [{tcycl_tc}] is {dist:.2f} Km away.")
                if dist < min_dist:
                    min_dist = dist
                    closest_tcycl = tcycl_tc
            
            if min_dist < threshold and closest_tcycl not in [p[1] for p in matched]:
                matched.append((trckr_tc, closest_tcycl))

        if len(matched):
            print(f"\n# {len(matched)} TC's matched")
        
        for tcr, tcp in matched:
            print("--------------")
            print(f"REAL: {tcr}")
            print(f"PRED: {tcp}")
            print("--------------")
        
        true_positives = [p[0] for p in matched]
        false_negatives = [tc for tc in ref_step_data if tc not in true_positives]
        
        tcycl_matched = [p[1] for p in matched]
        false_positives = [tc for tc in tcycl_data_step if tc not in tcycl_matched]
        
        # tp, fp, fn
        scores = {
            "true-positives": true_positives,
            "false-positives": false_positives,
            "false-negatives": false_negatives,
        }
        
        return scores
    
    
class TCPlotter(Scorer):
    """
    Compares and plots a TCYCL vs TRCKR predictions
    """

    def __init__(self,
                 trckr_pred,
                 tcycl_pred,
                 plot_steps_str,
                 output_dir,
                 ibtracks=None):
    
        self.trckr_prediction = trckr_pred
        self.tcycl_prediction = tcycl_pred
        self.plot_steps = [int(s) for s in plot_steps_str.split(",")]
        self.output_dir = output_dir
        self.ibtracks = ibtracks
    
        super().__init__()

    def run(self):
        """Run the actual plotting"""

        import matplotlib
        matplotlib.use('Agg')

        import matplotlib.pyplot as plt
        import matplotlib.colors as plt_cols
        
        this_dir = os.path.dirname(os.path.realpath(__file__))
        
        marker_size = 100
        trckr_cmap = plt.cm.get_cmap('plasma')
        p_min, p_max = 900, 1000
        
        # create output dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # make plots for each step
        for step in self.plot_steps:
            
            plt.figure(step, figsize=(12, 6))
            title_str = f"TC-Tracker and TCYCL data (step {step})"
            if self.ibtracks:
                title_str += " + IBtracks"
            plt.title(title_str)

            trckr_style = {"c": [tc.params["p"] for tc in self.trckr_prediction[step]],
                           "cmap": trckr_cmap,
                           "s": int(marker_size*2),
                           "marker": "o",
                           "vmin": p_min,
                           "vmax": p_max}
            
            tcycl_style = {"c": "k",
                           "s": marker_size,
                           "marker": "x"}

            ibtks_style = {"c": "k",
                           "s": marker_size,
                           "marker": "+",
                           "facecolors": "none"
                           }

            # background map
            bk_cmap = plt_cols.ListedColormap([[0.3, 0.4, 0.2, 0.8], [0.7, 0.7, 0.9, 0.9]])
            plt.imshow(np.load(os.path.join(this_dir, "map.npy")), extent=[0, 360, -90, 90], cmap=bk_cmap)
            
            plt.xlim([0, 360])
            plt.ylim([-90, 90])
            plt.grid()

            # plot TC's
            trckr_lats = [coords.lat for coords in self.trckr_prediction[step]]
            trckr_lons = [coords.lon for coords in self.trckr_prediction[step]]
            
            tcycl_lats = [coords.lat for coords in self.tcycl_prediction[step]]
            tcycl_lons = [coords.lon if coords.lon >= 0 else coords.lon+360
                          for coords in self.tcycl_prediction[step]]

            im_trckr = plt.scatter(trckr_lons, trckr_lats, label="TC-tracker", **trckr_style)
            _ = plt.scatter(tcycl_lons, tcycl_lats, label="TCYCL", **tcycl_style)
            
            if self.ibtracks:
                ibtks_lats = [coords.lat for coords in self.ibtracks[step]]
                ibtks_lons = [coords.lon if coords.lon >= 0 else coords.lon+360
                              for coords in self.ibtracks[step]]
                _ = plt.scatter(ibtks_lons, ibtks_lats, label="IB-Tracks", **ibtks_style)
            
            cbar = plt.colorbar(im_trckr, shrink=0.75)
            cbar.set_ticks(np.linspace(p_min, p_max, 6))
            cbar.set_ticklabels(np.linspace(p_min, p_max, 6))
            cbar.set_label("$P_{min}$")
            
            # axes
            plt.gca().set_aspect('equal')
            plt.xticks(list(range(0, 361, 40)))
            plt.xlabel("Lon")
            plt.yticks(list(range(-90, 91, 20)))
            plt.ylabel("Lat")
            
            plt.legend()
            
            save_path = os.path.join(self.output_dir, f"plot_step_{step:03}.png")
            
            print(f"Saving plot in {save_path}")
            plt.savefig(save_path)
            plt.close()
        
    def score(self):
        return None

    def summary(self):
        pass
    

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("trckr_summary_path", help=f"Path to CSV with Tracker summary")
    parser.add_argument("tcycl_summary_path", help=f"Path to CSV with Tcycl summary")
    
    parser.add_argument("--dist_threshold", help=f"TC distance threshold [Km]", type=int, default=300)
    
    parser.add_argument("--trckr_w_min", help=f"TC-Tracker TC's with wind speed >= threshold", type=float)
    parser.add_argument("--tcycl_w_min", help=f"TC-Tracker TC's with wind speed >= threshold", type=float)
    
    parser.add_argument("--plot_steps", help=f"Make plots for steps (comma-separated list - e.g. '6,12,18')")
    parser.add_argument("--output_dir", help=f"Directory where plots are saved", default="/var/tmp/plots")

    parser.add_argument("--ibtracks_summary_path", help=f"Path to IBTracks CSV summary")
    
    args = parser.parse_args()
    
    # # TC tracker and TCYCL data
    trckr_prediction = TRCKR_Forecast(args.trckr_summary_path, min_wind_speed=args.trckr_w_min)
    tcycl_prediction = TCYCL_Forecast(args.tcycl_summary_path, min_wind_speed=args.tcycl_w_min)

    # print confusion matrix (TC tracker and TCYCL data)
    scorer = ConfusionMatrix(trckr_prediction, tcycl_prediction, args.dist_threshold)
    scorer.run()
    scorer.summary()

    # Compare TCYCL VS IBTRACKS if requested
    ibtks_data = None
    if args.ibtracks_summary_path:
        
        ibtks_data = IBTKS_Data(args.ibtracks_summary_path, min_wind_speed=args.tcycl_w_min)

        # compare at first step only
        compare_step = tcycl_prediction.steps[0]
        scorer = ConfusionMatrix(ibtks_data,
                                 tcycl_prediction,
                                 args.dist_threshold,
                                 compare_at_steps=[compare_step])
        scorer.run()
        scorer.summary()

    # make plots
    if args.plot_steps:
        plotter = TCPlotter(trckr_prediction,
                            tcycl_prediction,
                            args.plot_steps,
                            args.output_dir,
                            ibtracks=ibtks_data)
        plotter.run()
        plotter.summary()


if __name__ == "__main__":
    sys.exit(main())