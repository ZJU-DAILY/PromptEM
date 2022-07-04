from openprompt import PromptForClassification, PromptDataLoader
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import PtuningTemplate, ManualVerbalizer
from tqdm import tqdm
from args import PromptEMArgs


def get_prompt_class_label_words(args:PromptEMArgs):
    classes = [
        "yes",
        "no",
    ]
    if args.template_no==4 or args.template_no==5:
        label_words = {
            "yes": ["Yes"],
            "no": ["No"],
        }
    elif args.one_word:
        label_words = {
            "yes": ["matched"],
            "no": ["mismatched"],
        }
    else:
        label_words = {
            "yes": ["matched", "similar", "relevant"],
            "no": ["mismatched", "different", "irrelevant"],
        }
    return classes, label_words


def get_prompt_components(args: PromptEMArgs):
    templates = [
        '{"placeholder":"text_a"} {"placeholder":"text_b"} {"soft": "They are "} {"mask"}',
        '{"placeholder":"text_a"} {"placeholder":"text_b"} They are {"mask"}',
        '{"placeholder":"text_a"} {"soft": "is"} {"mask"} {"soft": "to"} {"placeholder":"text_b"}',
        '{"placeholder":"text_a"} is {"mask"} to {"placeholder":"text_b"}',
        '{"placeholder":"text_a"} {"placeholder":"text_b"} {"soft": "Are they matched ? "} {"mask"}',
        '{"placeholder":"text_a"} {"placeholder":"text_b"} Are they matched ? {"mask"}',
    ]
    classes, label_words = get_prompt_class_label_words()
    plm, tokenizer, model_config, WrapperClass = load_plm(args.model_type, args.model_name_or_path)
    template = PtuningTemplate(
        model=plm,
        tokenizer=tokenizer,
        text=templates[args.template_no],
    )
    verbalizer = ManualVerbalizer(
        classes=classes,
        label_words=label_words,
        tokenizer=tokenizer,
    )
    return plm, tokenizer, WrapperClass, template, verbalizer


def get_prompt_model(args: PromptEMArgs):
    plm, tokenizer, wrapperClass, template, verbalizer = get_prompt_components(args)
    model = PromptForClassification(
        template=template,
        plm=plm,
        verbalizer=verbalizer,
    )
    return model, tokenizer, wrapperClass, template


def get_prompt_dataloader(args: PromptEMArgs, dataset, shuffle: bool, batch_size=None):
    plm, tokenizer, wrapperClass, template, verbalizer = get_prompt_components(args)
    return PromptDataLoader(
        dataset=dataset,
        tokenizer=tokenizer,
        template=template,
        tokenizer_wrapper_class=wrapperClass,
        batch_size=batch_size if batch_size else args.batch_size,
        max_seq_length=args.max_length,
        shuffle=shuffle,
    )


def read_prompt_dataset(left_entities, right_entities, pairs, y_truth):
    dataset = []
    for guid, pair in tqdm(enumerate(pairs), desc="generate prompt dataset..."):
        dataset.append(
            InputExample(
                guid=guid,
                text_a=left_entities[pair[0]],
                text_b=right_entities[pair[1]],
                label=y_truth[guid] if y_truth else 0
            )
        )
    return dataset
