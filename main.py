import argparse
from eval.eval_OpenNMT_KPG import recompute_scores_on_generated_kps

##########################################################
### ARGUMENTS

parser = argparse.ArgumentParser(description="""
For the sake of sanity check, this script loads the generated KPs and recompute the metrics.
It produces a TSV with the 
In the case of split and join, an additional processing can be applied on KPs (set or filter by threshold).
""", formatter_class=argparse.RawTextHelpFormatter)

parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
optional = parser.add_argument_group('optional arguments')
required.add_argument("-path_to_folder_with_candidate_kps", type=str, help="Directory where the TSV with the generated KPs (keyphrases) are saved")
required.add_argument("-output_kps_filename", type=str, help="Name of the TSV file with KPs")
optional.add_argument("-output_filename_prefix", default='recomputed', help="prefix added to new files with recomputes scores")
required.add_argument("-toy", required=True, type=int, help="1 for a quick test of the entire pipeline with only a few test instances, 0 for running the evaluation on all test instances")
required.add_argument("-preprocessing", type=str, default="stemming_mu", help="Apply some preprocessing on generated KPs", choices=['no_preprocessing', # give the candidate KPs as they are
                                                                                                             'merge_duplicate_kp', # apply a set on KPs
                                                                                                             'filter_stutter',
                                                                                                             'remove_all_stuttering', # remove the KPs that consist in the repetition of the same term
                                                                                                             'classic_en_stemming',
                                                                                                             'stemming_mu', # multilingual stemming, the processing used in the paper
                                                                                                             'custom_normalization',
                                                                                                             ])
args = parser.parse_args()




recompute_scores_on_generated_kps(path_to_folder_with_output_kps=args.path_to_folder_with_candidate_kps,
                                  output_kps_filename=args.output_kps_filename,
                                  output_filename_prefix=args.output_filename_prefix,
                                  toy=args.toy,
                                  preprocessing=args.preprocessing)
