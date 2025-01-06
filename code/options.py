from __future__ import absolute_import
from __future__ import division
import argparse
import uuid
import os
from pprint import pprint


def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_input_dir", default="", type=str)
    parser.add_argument("--input_file", default="train.txt", type=str)
    parser.add_argument("--create_vocab", default=0, type=int)
    parser.add_argument("--vocab_dir", default="", type=str)
    parser.add_argument("--max_num_actions", default=100, type=int)
    parser.add_argument("--path_length", default=3, type=int)
    parser.add_argument("--hidden_size", default=50, type=int)
    parser.add_argument("--embedding_size", default=50, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--grad_clip_norm", default=5, type=int)
    parser.add_argument("--l2_reg_const", default=1e-2, type=float)
    parser.add_argument("--learning_rate", default=1e-2, type=float)
    parser.add_argument("--beta", default=0.02, type=float)
    parser.add_argument("--alpha", default=0.2, type=float)
    parser.add_argument("--delta", default=0.1, type=float)

    parser.add_argument("--positive_reward", default=1.0, type=float)
    parser.add_argument("--negative_reward", default=0, type=float)
    parser.add_argument("--gamma", default=0.1, type=float)
    parser.add_argument("--log_dir", default="./logs/", type=str)
    parser.add_argument("--log_file_name", default="reward.txt", type=str)
    parser.add_argument("--output_file", default="", type=str)
    parser.add_argument("--num_rollouts", default=15, type=int)
    parser.add_argument("--test_rollouts", default=100, type=int)
    parser.add_argument("--LSTM_layers", default=3, type=int)
    parser.add_argument("--model_dir", default='', type=str)
    parser.add_argument("--base_output_dir", default='', type=str)
    parser.add_argument("--total_iterations", default=2000, type=int)
    parser.add_argument("--decay_batch", default=100, type=int)
    parser.add_argument("--decay_rate", default=0.9, type=float)

    parser.add_argument("--Lambda", default=0.02, type=float)
    parser.add_argument("--pool", default="max", type=str)
    parser.add_argument("--eval_every", default=100, type=int)
    parser.add_argument("--use_entity_embeddings", default=0, type=int)
    parser.add_argument("--use_cluster_embeddings", default=1, type=int)
    parser.add_argument("--train_entity_embeddings", default=0, type=int)
    parser.add_argument("--train_relation_embeddings", default=1, type=int)
    parser.add_argument("--model_load_dir", default="", type=str)
    parser.add_argument("--load_model", default=0, type=int)
    parser.add_argument("--nell_evaluation", default=0, type=int)
    parser.add_argument("--use_cuda", default=1, type=int)

    try:
        parsed = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))
    parsed['input_files'] = [parsed['data_input_dir'] + '/' + parsed['input_file']]

    parsed['use_entity_embeddings'] = (parsed['use_entity_embeddings'] == 1)
    parsed['train_entity_embeddings'] = (parsed['train_entity_embeddings'] == 1)
    parsed['train_relation_embeddings'] = (parsed['train_relation_embeddings'] == 1)

    parsed['output_dir'] = parsed['base_output_dir'] + '/' + str(uuid.uuid4())[:4]+'_'+str(parsed['path_length'])+'_'+str(parsed['beta'])+'_'+str(parsed['test_rollouts'])+'_'+str(parsed['Lambda'])

    parsed['model_dir'] = parsed['output_dir']+'/'+ 'model/'
    parsed['reward_dir'] = parsed['output_dir'] + '/' + 'reward.npy'

    parsed['load_model'] = (parsed['load_model'] == 1)

    ##Logger##
    parsed['path_logger_file'] = parsed['output_dir']
    parsed['log_file_name'] = parsed['output_dir'] +'/log.txt'
    os.makedirs(parsed['output_dir'])
    os.mkdir(parsed['model_dir'])
    with open(parsed['output_dir']+'/config.txt', 'w') as out:
        pprint(parsed, stream=out)

    # print and return
    maxLen = max([len(ii) for ii in parsed.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(parsed.items()): print(fmtString % keyPair)
    return parsed
