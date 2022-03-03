import argparse

from tasks import ki_mlkg

def main_func(args):
    ki_mlkg(args)  # MLKG integration (pretraining)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual Space")

    # data
    parser.add_argument("--data_dir", type=str, default="/cluster/work/sachan/yifan/data/wikidata/sub_clean_sample20",
                        help="the input data directory.")
    parser.add_argument("--model_dir", type=str, default="/cluster/work/sachan/yifan/huggingface_models/bert-base-multilingual-cased",
                        help="The stored model directory.")
    parser.add_argument("--tmp_dir", type=str, default="./tmp/checkpoints",
                        help="The stored model directory.")

    # model
    parser.add_argument("--device", type=int, default=-1,
                        help="which GPU to use. set -1 to use CPU.")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="learning rate of GCS.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-6,
                        help="Adam epsilon")
    parser.add_argument("--warmup_steps", type=int, default=1e4,
                        help="number of training epochs.")
    parser.add_argument("--entity_epoch", type=int, default=5,
                        help="number of training epochs.")
    parser.add_argument("--triple_epoch", type=int, default=10,
                        help="number of training epochs.")
    parser.add_argument("--patience", type=int, default=10,
                        help="used for early stop")
    parser.add_argument("--batch_num", type=int, default=32,
                        help="number of triple samples per 1 batch")
    parser.add_argument("--neg_num", type=int, default=8,
                        help="number of negative samples")
    parser.add_argument("--lm_mask_token_id", type=int, default=0,
                        help="token id of masked token")
    parser.add_argument("--adapter_num", type=int, default=8,
                        help="number of different adapters")


    args = parser.parse_args()
    print(args)
    main_func(args)