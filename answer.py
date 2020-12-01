import transformers
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering, XLNetTokenizer, XLNetForQuestionAnsweringSimple
import torch
import os
import torch
import time
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import (
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    squad_convert_examples_to_features
)

from transformers import (
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer
)

from transformers.data.processors.squad import SquadResult, SquadV2Processor, SquadExample

from transformers.data.metrics.squad_metrics import compute_predictions_logits



def answergen(context, question):

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

    encoding = tokenizer.encode_plus(question, context)

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    start_scores, end_scores = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))

    ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)

    print ("\nQuestion ",question)
    #print ("\nAnswer Tokens: ")
    #print (answer_tokens)

    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
    #print ("\nAnswer : ",answer_tokens_to_string)
    return answer_tokens_to_string

def answergen_xlnet(context, question):
    tokenizer = XLNetTokenizer.from_pretrained('ahotrod/xlnet_large_squad2_512 ')
    model = XLNetForQuestionAnsweringSimple.from_pretrained('ahotrod/xlnet_large_squad2_512 ')
    #tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad',return_token_type_ids = True)
    #model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    encoding = tokenizer.encode_plus(question, context)

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    start_scores, end_scores = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))

    ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens)

    print ("\nQuestion ",question)
    #print ("\nAnswer Tokens: ")
    #print (answer_tokens)

    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
    print ("\nAnswer : ",answer_tokens_to_string)
    return answer_tokens_to_string


def answergen_bert(context, question):
    tokenizer = BertTokenizer.from_pretrained('csarron/bert-base-uncased-squad-v1')
    model = BertForQuestionAnswering.from_pretrained('csarron/bert-base-uncased-squad-v1')
    #tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad',return_token_type_ids = True)
    #model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    encoding = tokenizer.encode_plus(question, context)

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    start_scores, end_scores = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))

    ans_tokens = input_ids[torch.argmax(start_scores[0, 1:]) : torch.argmax(end_scores[0, 1:])+1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens)

    print ("\nQuestion ",question)
    #print ("\nAnswer Tokens: ")
    #print (answer_tokens)

    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
    print ("\nAnswer : ",answer_tokens_to_string)
    return answer_tokens_to_string

def answergen_xlm(context, question):
    tokenizer = AutoTokenizer.from_pretrained('mrm8488/longformer-base-4096-finetuned-squadv2')
    model = AutoModelForQuestionAnswering.from_pretrained('mrm8488/longformer-base-4096-finetuned-squadv2')
    #tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad',return_token_type_ids = True)
    #model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    encoding = tokenizer.encode_plus(question, context)

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    start_scores, end_scores = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))

    ans_tokens = input_ids[torch.argmax(start_scores[0, 1:]) : torch.argmax(end_scores[0, 1:])+1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens)

    print ("\nQuestion ",question)
    #print ("\nAnswer Tokens: ")
    #print (answer_tokens)

    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
    print ("\nAnswer : ",answer_tokens_to_string)
    return answer_tokens_to_string

def answergen_albert(question_texts, context_text):
    model_name_or_path = "ktrapeznikov/albert-xlarge-v2-squad-v2"
    output_dir = ""
    # Config
    n_best_size = 1
    max_answer_length = 30
    do_lower_case = True
    null_score_diff_threshold = 0.0

    def to_list(tensor):
        return tensor.detach().cpu().tolist()

    # Setup model
    config_class, model_class, tokenizer_class = (
        AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer)
    config = config_class.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")
    tokenizer = tokenizer_class.from_pretrained(
        "ktrapeznikov/albert-xlarge-v2-squad-v2", do_lower_case=True)
    model = model_class.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    #processor = SquadV2Processor()
    """Setup function to compute predictions"""
    examples = []
    print(question_texts)
    for i, question_text in enumerate(question_texts):
        example = SquadExample(
            qas_id=str(i),
            question_text=question_text,
            context_text=context_text,
            answer_text=None,
            start_position_character=None,
            title="Predict",
            is_impossible=False,
            answers=None,
        )

        examples.append(example)
    print(examples)
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=False,
        return_dataset="pt",
        threads=1,
    )

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=10)

    all_results = []

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]

            outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

    output_prediction_file = "predictions.json"
    output_nbest_file = "nbest_predictions.json"
    output_null_log_odds_file = "null_predictions.json"

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        False,  # verbose_logging
        True,  # version_2_with_negative
        null_score_diff_threshold,
        tokenizer,
    )

    return predictions
