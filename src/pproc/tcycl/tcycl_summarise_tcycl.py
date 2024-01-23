import sys
import csv
import json
import argparse


class SummaryWriter:
    """
    Combine TCycl predictions and annotations in a
    human-readable Summary format (e.g. CSV file)
    """
    
    fields_base = [
        "name",
        "step",
        "lat",
        "lon",
    ]
    
    fields_annot = [
        "P_min",
        "P_min_lat",
        "P_min_lon",
        "V10_max",
        "V10_max_lat",
        "V10_max_lon",
    ]
    
    def __init__(self, prediction_files, annotation_files=None):
        """
        Writes a easily readable CSV summary with Tcycl prediction data
        """

        print("Parsing data files..")

        # prediction data
        self.prediction_titles = self.fields_base

        predictions_files_dict = {int(pred_name.split(".")[1].split("step")[1]): pred_name
                                  for pred_name in prediction_files.split(";")}

        # annotation data (if available)
        annotations_files_dict = {}
        if annotation_files is not None:
            self.prediction_titles += self.fields_annot
            annotations_files_dict = {int(anno_name.split(".")[1].split("step")[1]): anno_name
                                      for anno_name in annotation_files.split(";")}

        # compose overall data structure
        self.prediction_data = {}
        for step, pred_name in predictions_files_dict.items():
            
            with open(pred_name, "r") as f:
                prediction_content = json.load(f)
                
            if annotation_files is not None:
                with open(annotations_files_dict[step], "r") as f:
                    annotation_content = json.load(f)
                assert len(annotation_content) == len(prediction_content)

            step_content = []
            for tc_idx, tc in enumerate(prediction_content):
                tc_content = ["unnamed", step, *tc]
                if annotation_files is not None:
                    tc_content.extend(annotation_content[tc_idx])
                step_content.append(tc_content)
    
            # step: [name, step, lat, lon]
            self.prediction_data.update({step: step_content})

    def write(self, output=None):
        """
        Write summary output
        """
        
        print(f"Writing output CSV file {output}")
        
        with open(output, 'w') as f:
            _writer = csv.writer(f)
            _writer.writerow(self.prediction_titles)
            for step, pred in self.prediction_data.items():
                for tc in pred:
                    _writer.writerow(tc)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_files", help=f";-separated list of prediction files (as prediction.step<N>.json)")
    parser.add_argument("--annotation_files", help=f";-separated list of annotation files (as annotation.step<N>.json)")
    parser.add_argument("--output", help=f"Path to output CSV file", default="output.csv")
    args = parser.parse_args()
    
    # Make summary file
    writer = SummaryWriter(args.prediction_files, annotation_files=args.annotation_files)
    writer.write(output=args.output)


if __name__ == "__main__":
    sys.exit(main())
