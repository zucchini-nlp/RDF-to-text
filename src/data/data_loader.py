from transformers.tokenization_utils import PreTrainedTokenizer

from .data_utils import make_data_module
from .sft_dataset import DataCollatorForSupervisedDataset, SFTInstructionDataset


def make_supervised_data_module(tokenizer: PreTrainedTokenizer, args):
    train_dataset, eval_dataset = make_data_module(args)
    max_seq_length = tokenizer.model_max_length

    train_dataset = (
        SFTInstructionDataset(
            train_dataset,
            tokenizer=tokenizer,
            max_seq_len=max_seq_length,
        )
        if args.do_train
        else None
    )

    eval_dataset = (
        SFTInstructionDataset(
            eval_dataset,
            tokenizer=tokenizer,
            max_seq_len=max_seq_length,
        )
        if args.do_eval
        else None
    )

    print(f"train_dataset length: {len(train_dataset)}") if args.do_train else None
    print(f"eval_dataset length: {len(eval_dataset)}" ) if args.do_eval else None
    print("Adding data collator: ", DataCollatorForSupervisedDataset)
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, predict_with_generate=args.predict_with_generate
    )

    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator,
    )



def make_prediction_dataset(tokenizer: PreTrainedTokenizer, args):
    dataset = make_data_module(args, split=False)

    if 'eval' in dataset.column_names:
        eval_dataset = dataset['eval']
    else:
        eval_dataset = dataset['train']

    inputs, outputs = [], []
    for element in eval_dataset:
        inputs.append(element['input'])
        outputs.append(element['output'])

    return inputs, outputs