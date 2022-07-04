import argparse
import logging


class PromptEMArgs:
    def __init__(self, args, data_name: str) -> None:
        self.seed = args.seed
        self.device = args.device
        self.model_name_or_path = args.model_name_or_path
        self.model_type = self.model_name_or_path.split("/")[-1].split("-")[0]
        self.batch_size = args.batch_size
        self.text_summarize = args.text_summarize
        self.learning_rate = args.lr
        self.max_length = args.max_length
        self.add_token = args.add_token
        self.data_name = data_name
        self.template_no = args.template_no
        self.self_training = args.self_training
        self.dynamic_dataset = args.dynamic_dataset
        self.k = args.k
        self.num_iter = args.num_iter
        self.pseudo_label_method = args.pseudo_label_method
        self.confidence_ratio = args.confidence_ratio
        self.mc_dropout_pass = args.mc_dropout_pass
        self.uncertainty_ratio = args.uncertainty_ratio
        self.el2n_ratio = args.el2n_ratio
        self.save_model = args.save_model
        self.only_plm = args.only_plm
        self.teacher_epochs = args.teacher_epochs
        self.student_epochs = args.student_epochs
        self.test_pseudo_label = args.test_pseudo_label
        self.one_word=args.one_word
        if self.dynamic_dataset != -1:
            assert self.self_training
        if not self.self_training and len(args.test_pseudo_label) == 0:
            self.teacher_epochs *= 2
        if "text" in self.data_name and not self.only_plm:
            self.text_summarize = True

    def __str__(self) -> str:
        return f"[{', '.join((f'{k}:{v}' for (k, v) in self.__dict__.items()))}]"

    def log(self):
        logging.info("====PromptEM Args====")
        for (k, v) in self.__dict__.items():
            logging.info(f"{k}: {v}")


def int_or_float(value):
    try:
        value = int(value)
        return value
    except ValueError:
        try:
            value = float(value)
            return value
        except ValueError:
            return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--batch_size", "-bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5, help="(teacher) lr")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--add_token", default=True)
    parser.add_argument("--data_name", "-d", type=str,
                        choices=["rel-heter", "rel-text", "semi-heter", "semi-homo", "semi-rel", "semi-text-c",
                                 "semi-text-w", "all"], default="all")
    parser.add_argument("--template_no", "-tn", type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--self_training", "-st", action="store_true", default=False)
    parser.add_argument("--dynamic_dataset", "-dd", type=int, default=-1,
                        help="-1 means that dd is off, otherwise it means the frequency of dd.")
    parser.add_argument("--num_iter", "-ni", type=int, default=1)
    parser.add_argument("--k", "-k", type=int_or_float, default=0.05)
    parser.add_argument("--pseudo_label_method", "-pm", type=str, default="uncertainty",
                        choices=["uncertainty", "confidence", "unfold_fold"])
    parser.add_argument("--mc_dropout_pass", "-mdp", type=int, default=10)
    parser.add_argument("--uncertainty_ratio", "-ur", type=float, default=0.1)
    parser.add_argument("--el2n_ratio", "-er", type=float, default=0.1)
    parser.add_argument("--confidence_ratio", "-cr", type=float, default=0.1)
    parser.add_argument("--text_summarize", "-ts", action="store_true")
    parser.add_argument("--save_model", "-save", action="store_true", default=False)
    parser.add_argument("--only_plm", "-op", action="store_true", default=False)
    parser.add_argument("--teacher_epochs", "-te", type=int, default=20)
    parser.add_argument("--student_epochs", "-se", type=int, default=30)
    parser.add_argument("--test_pseudo_label", "-tpl", type=str, default="")
    parser.add_argument("--one_word","-ow",action="store_true",default=False)

    args = parser.parse_args()
    return args


def parse_em_args(args, data_name) -> PromptEMArgs:
    return PromptEMArgs(args, data_name)
