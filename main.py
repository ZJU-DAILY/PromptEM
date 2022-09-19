import logging
from args import parse_args, parse_em_args
from data import PromptEMData
from train import self_training, self_training_only_plm
from utils import set_seed, set_logger, read_entities, read_ground_truth_few_shot, read_ground_truth

if __name__ == '__main__':
    common_args = parse_args()
    set_logger("PromptEM")
    tasks = [common_args.data_name]
    if common_args.data_name == "all":
        tasks = ["rel-heter", "rel-text", "semi-heter", "semi-homo", "semi-rel", "semi-text-c", "semi-text-w", "geo-heter"]
    for data_type in tasks:
        # args and global data
        args = parse_em_args(common_args, data_type)
        args.log()
        data = PromptEMData(data_type)
        # entities
        data.left_entities, data.right_entities = read_entities(data_type, args)
        # samples
        data.train_pairs, data.train_y, \
        data.train_un_pairs, data.train_un_y = read_ground_truth_few_shot(f"data/{data_type}", ["train"], k=args.k,
                                                                          return_un_y=True)
        data.valid_pairs, data.valid_y = read_ground_truth(f"data/{data_type}", ["valid"])
        data.test_pairs, data.test_y = read_ground_truth(f"data/{data_type}", ["test"])
        logging.info(f"left size: {len(data.left_entities)}, right size: {len(data.right_entities)}")
        logging.info(f"labeled train size: {len(data.train_pairs)}")
        logging.info(f"unlabeled train size: {len(data.train_un_pairs)}")
        logging.info(f"valid size: {len(data.valid_pairs)}")
        logging.info(f"test size: {len(data.test_pairs)}")
        # for checking pseudo label acc
        data.read_all_ground_truth(f"data/{data_type}")
        set_seed(common_args.seed)
        if args.only_plm:
            self_training_only_plm(args, data)
        else:
            self_training(args, data)
