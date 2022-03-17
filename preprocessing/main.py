import argparse

from preprocessing import preprocess_clean, preprocess_pre, pre_cooccurrence, preprocess_rlabel, preprocess_des
from preexp import pre_static_dist, pre_mapping_dist


def main_func(args):
    # preprocess_rlabel(args)
    preprocess_clean(args, "small")  # clean wikidata entities
    # preprocess_des(args)  # get descriptions for entities
    # preprocess_pre(args)  # clean a subset of wikidata entities for pre-experiment
    # pre_static_dist(args)  # calculate entity label embedding (store in ./embed)
    # pre_mapping_dist(args)
    # pre_cooccurrence(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual Space")

    # data
    parser.add_argument("--data_dir", type=str, default="/cluster/work/sachan/yifan/data/wikidata/sub_clean_rich10",
                        help="the input data directory.")
    parser.add_argument("--file_idx", type=str, default="000",
                        help="the idx of the entity file in parallel.")
    parser.add_argument("--model_dir", type=str, default="/cluster/work/sachan/yifan/huggingface_models/",
                        help="The stored model directory.")
    parser.add_argument("--simulate_model", type=str, default="xlm-roberta-base",
                        help="multilingual LMs to analyze")
    parser.add_argument("--mapping_model", type=str, default="1_layer",
                        help="multilingual LMs to analyze")
    parser.add_argument("--data_type", type=str, default="labels",
                        help="multilingual LMs to analyze")

    # model
    parser.add_argument("--device", type=int, default=-1,
                        help="which GPU to use. set -1 to use CPU.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate of GCS.")
    parser.add_argument("--epoch", type=int, default=1000,
                        help="number of training epochs.")
    parser.add_argument("--patience", type=int, default=10,
                        help="used for early stop")

    args = parser.parse_args()
    print(args)
    main_func(args)