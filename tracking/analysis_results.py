import argparse
import os
import sys
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if env_path not in sys.path:
    sys.path.append(env_path)
def anaylsis(dataset_name, tracker_name, tracker_param):
    from lib.test.analysis.plot_results import print_results
    from lib.test.evaluation import get_dataset, trackerlist

    trackers = []


    trackers.extend(trackerlist(name=tracker_name, parameter_name=tracker_param, dataset_name=dataset_name,
                                run_ids=None, display_name=tracker_param))

    dataset = get_dataset(dataset_name)

    print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'),
                  force_evaluation=True)

    # Available datasets include uav, nfs, lasot_extension_subset, lasot, and tnl2k.
def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--dataset_name', type=str, default='lasot', help='Name of dataset (nfs, uav, got10k_test, '
                                                                          'got10k_val, got10k_ltrval, lasot, trackingnet, '
                                                                          'lasot_extension_subset, tnl2k).')

    args = parser.parse_args()



    anaylsis(args.dataset_name, args.tracker_name, args.tracker_param)


if __name__ == '__main__':
    main()
