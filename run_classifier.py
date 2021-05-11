# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import annotations, absolute_import

import os
from typing import Dict, List


from transformers import (
    AutoTokenizer, BertTokenizer, 
    BertForSequenceClassification, BertConfig,
    Trainer, TrainingArguments,
    PreTrainedTokenizer
)
from transformers.configuration_utils import PretrainedConfig

from src.schema import (
    InputExample, InputFeatures, Config
)
from src.data_process import (
    AgNewsDataProcessor
)

from config import create_logger

logger = create_logger()



def convert_single_example(
        example_index: int, example: InputExample, label2id: Dict[str, int], max_seq_length: int, tokenizer: BertTokenizer
) -> InputFeatures:
    """Converts a single `InputExample` into a single `InputFeatures`.

    example_index: 用于展示example中的前几例数据
    """
    parameters  = {
        "text":example.text_a,
        "add_special_tokens":True,
        "padding":True,
        "max_length":max_seq_length,
        "return_attention_mask":True,
        "return_token_type_ids":True,
        "return_length":True,
        "verbose":True
    }
    if example.text_b:
        parameters['text_pair'] = example.text_b
    feature = tokenizer(**parameters)

    input_feature = InputFeatures(
        input_ids=feature['token_ids'],
        attention_mask=feature['attention_mask'],
        segment_ids=feature['token_type_ids'],
        label_id=label2id[example.label],
        is_real_example=True
    )
    
    if example_index < 5:
        logger.info(f'*************************** Example {example_index} ***************************')
        logger.info(example)
        logger.info(input_feature)
        logger.info('*************************** Example End ***************************')
    
    return input_feature


def create_bert_for_sequence_classification_model(config: Config):
    bert_config: BertConfig = BertConfig.from_pretrained(config.pretrained_model_name)
    bert_config.num_labels = config.num_labels
    model = BertForSequenceClassification(bert_config)
    return model

def create_model(config: Config):
    """Creates a classification model."""
    models = {
        "bert-for-sequence-classification": create_bert_for_sequence_classification_model,
    }
    return models[config.model_name](config)

def convert_examples_to_features(
        examples, label_list: List[str], 
        max_seq_length: int, tokenizer: PreTrainedTokenizer
    ):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    label2id = {label: index for index, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 200 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label2id,
                                         max_seq_length, tokenizer)
        features.append(feature)
    return features


class SequenceClassificationTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        return outputs.loss

def main():

    # processors need to be updated
    processors = {
        'agnews-processor': AgNewsDataProcessor,
    }

    config: Config = Config.instance()

    if not config.do_train and not config.do_eval and not config.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = PretrainedConfig.from_pretrained(config.pretrained_model_name)
    # 根据不同的任务，处理不同的数据集
    task_name = config.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if config.do_train:

        train_examples: List[InputExample] = processor.get_train_examples(config.data_dir)
        train_dataset_loader = 
        num_train_steps = int(
            len(train_examples) / config.train_batch_size * config.epochs
        )
        num_warmup_steps = int(num_train_steps * config.warmup_proportion)
        
        model = create_model(config=config)
        training_arguments = TrainingArguments(
            output_dir=config.output_dir,
            overwrite_output_dir=True,
        )
        trainer = SequenceClassificationTrainer(
            model=model,
            
        )

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPUs
    

    if config.do_train:
        train_file = os.path.join(config.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, config.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", config.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=config.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if config.do_eval:
        eval_examples = processor.get_dev_examples(config.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if config.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % config.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(config.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, config.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", config.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if config.use_tpu:
            assert len(eval_examples) % config.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // config.eval_batch_size)

        eval_drop_remainder = True if config.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=config.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(config.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if config.do_predict:
        predict_examples = processor.get_test_examples(config.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if config.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % config.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(config.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                config.max_seq_length, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", config.predict_batch_size)

        predict_drop_remainder = True if config.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=config.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(config.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
    main()
