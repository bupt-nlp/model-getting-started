from __future__ import annotations
import datetime
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

from src.data_process import THCNewsDataProcessor, DataIterator
from src.models.base_model import SequenceClassificationModel, BiLSTMSequenceClassificationModel
from src.schema import Config
from src.utils import get_time_dif

def create_model_config() -> Config:
    config = Config()
    # set data path, contains: train and test file
    root = os.path.join(os.path.abspath('.'), 'data')
    config.datadir = os.path.join(root, 'THCNews')
    config.language = 'zh'
    # pretrained model path, contains:
    # 1. pretrained model's binary file
    # 2. vocab
    pretrained_path = os.path.join(os.path.join(root, 'pretrained'), config.language)
    config.vocab_file = os.path.join(pretrained_path, 'vocab.pkl')
    config.pretrained_model_name = 'embedding_SougouNews.npz'
    config.pretrained_file = os.path.join(pretrained_path, config.pretrained_model_name)
    # save log with time here
    config.log_dir = os.path.join(config.datadir, 'log')
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    # save model after training here
    config.output_dir = os.path.join(config.datadir, 'save_dict')
    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)
    # load pretrained model
    config.pretrained_model = torch.tensor(
        np.load(config.pretrained_file)["embeddings"].astype('float32'))
    config.embedding_dim = config.pretrained_model.size()[1]
    config.hidden_size = 128
    config.num_layers = 2
    config.dropout = 0.5
    config.num_labels = 10
    config.max_seq_length = 40
    config.num_epochs = 8
    config.class_list = []
    with open(os.path.join(config.datadir, 'class.txt')) as  f:
        lines = f.readlines()
        for line in lines:
            config.class_list.append(line.strip())
    return config


def create_sequence_classification_model(config: Config) -> SequenceClassificationModel:
    model = BiLSTMSequenceClassificationModel(config)
    # 初始化
    return model


def train(config, model, train_iterator, dev_iterator, test_iterator):
    # record start running time
    start_time = time.time()
    dev_best_loss: float = float('inf')
    # last batch number which improves the result
    last_improve: float = 0
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate)
    # num of training batches
    batches: int = 0
    # if early stop is True, stop training
    early_stop = False

    for epoch in range(config.epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for (train_x, train_y) in train_iterator.get_batch():
            # set model to train model
            model.train()
            # transform data to current device
            train_x = train_x.to(config.device)
            train_y = train_y.to(config.device)
            output = model(train_x)
            model.zero_grad()
            # TODO: Implement a universial loss function to fit different task
            loss = F.cross_entropy(output, train_y.view(-1))
            loss.backward()
            optimizer.step()

            if batches % config.save_checkpoints_steps == 0:
                true = train_y.data.cpu()
                predict = torch.max(output.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predict)
                dev_acc, dev_loss = eval(model, dev_iterator, config, do_test=False)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    t = datetime.datetime.now()
                    state_path = os.path.join(config.output_dir,
                                              f'{t.year}-{t.month}-{t.day}-{t.hour}-lstm-classifer.ckpt')
                    torch.save(model.state_dict(), state_path)
                    improve = "*"
                    last_improve = batches
                else:
                    improve = ''

                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(batches, loss.item(), train_acc, dev_loss, dev_acc,
                                     get_time_dif(start_time), improve))
                if batches - last_improve > config.require_improve:
                    early_stop = True
                    print('No improving for a long time')
                    break
            batches += 1
            if early_stop:
                break
    test(model, test_iterator, config, model_file=state_path)


def eval(model: SequenceClassificationModel,
         dev_iterator: DataIterator,
         config,
         do_test=False) -> (float, float):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    num_batches: int = 0
    # evaluating model, don't update parameters
    with torch.no_grad():
        for texts, labels in dev_iterator.get_batch():
            texts = texts.to(config.device)
            labels = labels.to(config.device)
            outputs = model(texts)
            # TODO: Implement a universal loss function to replace the code
            loss = F.cross_entropy(outputs, labels.view(-1))
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)
            num_batches += 1

    # TODO: Replace these code with Metrics Reporter
    acc = metrics.accuracy_score(labels_all, predict_all)
    if do_test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / num_batches, report, confusion
    return acc, loss_total / num_batches


def test(model: BiLSTMSequenceClassificationModel,
         test_iterator: DataIterator,
         config: Config,
         model_file='data/THCNews/save_dict/2021-5-13-0-lstm-classifer.ckpt'):
    model.load_state_dict(torch.load(model_file))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = eval(model, test_iterator, config, do_test=True)
    msg = "Test Loss: {0:>5.2}, Test Acc: {1:>6.2%}"
    print(msg.format(test_loss, test_acc))
    time_dif = get_time_dif(start_time)
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    config = create_model_config()
    # config = config.parse_args(known_only=True)

    # 0. Load vocab
    vocab = pickle.load(open(config.vocab_file, 'rb'))
    config.n_vocab = len(vocab)

    # 2. Load data
    print('*' * 20, 'Loading data', '*' * 20)
    data_processor = THCNewsDataProcessor()
    root = os.path.join(os.path.abspath('.'), 'data')
    thcnews = os.path.join(root, 'THCNews')
    train_file = os.path.join(thcnews, 'train.txt')
    train_features = data_processor.get_train_features(vocab, train_file)
    dev_file = os.path.join(thcnews, 'dev.txt')
    dev_features = data_processor.get_dev_features(vocab, dev_file)
    test_file = os.path.join(thcnews, 'test.txt')
    test_features = data_processor.get_test_features(vocab, test_file)
    print(f'Trainset Length: {len(train_features)}, First example: {train_features[0]}')
    print(f'Dev Length: {len(dev_features)}, First example: {dev_features[0]}')
    print(f'Testset Length: {len(test_features)}, First example: {test_features[0]}')
    print('*' * 20, '   Loaded!  ', '*' * 20)

    # get iterator
    train_iterator = DataIterator(train_features,
                                  batch_size=config.train_batch_size,
                                  max_len=config.max_seq_length,
                                  padding_fill=len(vocab) - 1)
    dev_iterator = DataIterator(dev_features,
                                batch_size=config.eval_batch_size,
                                max_len=config.max_seq_length,
                                padding_fill=len(vocab) - 1)
    test_iterator = DataIterator(test_features,
                                batch_size=config.eval_batch_size,
                                max_len=config.max_seq_length,
                                padding_fill=len(vocab) - 1)

    model = create_sequence_classification_model(config)
    print(model)
    model = model.to(config.device)
    train(config, model, train_iterator, dev_iterator, test_iterator)
    test(model, test_iterator, config)