from __future__ import absolute_import
from __future__ import division
import numpy as np
from code.data.feed_data import RelationEntityAndClusterBatcher
from code.data.grapher import RelationEntityGrapher, RelationClusterGrapher
import logging
import torch

logger = logging.getLogger()
def calc_cluster_embedding(pretrained_entity_embeddings,entity_id_to_cluster_mappping):
    num_entity = pretrained_entity_embeddings.shape[0]  # the numbers of entity
    dim_emb = pretrained_entity_embeddings.shape[1]  # the dimension of entity embedding
    start_index = 10
    max_cluster = 0
    for i in range(start_index, num_entity):
        max_cluster = max(max_cluster, entity_id_to_cluster_mappping[
            str(i)])
    mapping_dict = {}
    count_dict = {}

    for i in range(max_cluster + 1):
        mapping_dict[str(i)] = np.zeros(dim_emb, dtype=np.float64)
        count_dict[str(i)] = 0

    for i in range(start_index, num_entity):
        mapping_dict[str(entity_id_to_cluster_mappping[str(i)])] = np.add(
            mapping_dict[str(entity_id_to_cluster_mappping[str(i)])],
            pretrained_entity_embeddings[i].cpu())
        count_dict[str(entity_id_to_cluster_mappping[str(i)])] += 1

    for i in range(max_cluster + 1):
        if count_dict[str(i)] == 0:
            continue
        mapping_dict[str(i)] = np.divide(mapping_dict[str(i)], count_dict[str(i)])

    return mapping_dict

class Efficient_Guidance_Exploration:
    def __init__(self, size, delta, epsilon):
        self.delta = [delta] * size
        self.epsilon = epsilon
        self.lam = [0.1] * size
        self.alpha = 1e-3
        self.size = size

    def update_lam(self, cluster_reward, embedding_cosine):
        for i in range(self.size):
            bound = self.delta[i] / (cluster_reward[i] + self.epsilon)
            if embedding_cosine[i] > bound:
                self.lam[i] -= self.alpha / (1 - self.lam[i])
            else:
                self.lam[i] += self.alpha / self.lam[i]

    def get_lam(self):
        return self.lam

class EntityEpisode(object):

    def __init__(self, graph, entity_data, cluster_data, params, mode):
        self.avg_cos_embb = 0
        self.grapher = graph

        self.pretrained_entity_embeddings = params['pretrained_embeddings_entity']
        self.pretrained_relation_embeddings = params['pretrained_embeddings_relation']


        self.mode = mode
        if self.mode == 'train':
            self.num_rollouts = params['num_rollouts']
        else:
            self.num_rollouts = params['test_rollouts']

        self.batch_size = params['batch_size']
        self.positive_reward = params['positive_reward'] # reward = 1
        self.negative_reward = params['negative_reward'] # reward = 0

        self.path_len = params['path_length']
        self.test_rollouts = params['test_rollouts']
        self.alpha = params['alpha']
        self.delta = params['delta']

        self.entity_id_to_cluster_mappping = params['entity_id_to_cluster_mappping'] #  map
        self.mapping_dict = calc_cluster_embedding(self.pretrained_entity_embeddings,self.entity_id_to_cluster_mappping) # mapping dict

        self.current_hop = 0
        start_entities, query_relation,  end_entities, all_entity_answers = entity_data

        start_clusters, end_clusters, all_cluster_answers = cluster_data

        start_clusters = np.repeat(start_clusters, self.num_rollouts)
        end_clusters = np.repeat(end_clusters, self.num_rollouts)  # [cls1, cls1, ..., cls2, cls2, ..., cls3, cls3, ...]
        # share experience

        self.no_examples = start_entities.shape[0] # original batch_size

        start_entities = np.repeat(start_entities, self.num_rollouts)
        batch_query_relation = np.repeat(query_relation, self.num_rollouts)
        end_entities = np.repeat(end_entities, self.num_rollouts) # [ent1, ent1, ..., ent2, ent2, ..., ent3, ent3, ...]

        self.cluster_path = {}
        self.approximated_reward = {}
        self.e_agent_cls_path = {}
        for i, ent in enumerate(start_entities):
            self.cluster_path[i] = [self.entity_id_to_cluster_mappping[str(ent)]]
            self.approximated_reward[i] = []
        for p_len in range(self.path_len):
            self.e_agent_cls_path[p_len] = []
        self.credits = []

        self.start_entities = start_entities
        self.end_entities = end_entities
        self.current_entities = start_entities
        self.query_relation = batch_query_relation
        self.all_entity_answers = all_entity_answers
        self.all_cluster_answers = all_cluster_answers
        self.epsilon = 0.01

        self.start_clusters = start_clusters
        self.end_clusters = end_clusters
        self.current_clusters = start_clusters

        next_actions = self.grapher.init_actions(self.current_entities, self.start_entities, self.query_relation,
                                                            self.end_entities, self.all_entity_answers, self.current_hop == self.path_len - 1,
                                                            self.num_rollouts)

        self.state = {}
        self.state['next_relations'] = next_actions[:, :, 1] # shape: [original batch_size * num_rollout, max_num_actions]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities
        self.EGP = Efficient_Guidance_Exploration(self.current_entities.shape[0],self.delta,self.epsilon)

    def get_state(self):
        return self.state

    def get_query_relation(self):
        return self.query_relation

    def get_reward(self):
        reward = (self.current_entities == self.end_entities)
        reward_cluster = (self.current_clusters == self.end_clusters)
        # set the True and False values to the values of positive and negative rewards.
        condlist = [reward == True, reward == False]
        choicelist = [self.positive_reward, self.negative_reward]
        # default reward
        reward_1 = np.select(condlist, choicelist)  # [original batch_size * num_rollout]

        condlist_cluster = [reward_cluster == True, reward_cluster == False]
        oper = np.select(condlist_cluster, choicelist)
        # cosine reward
        embedding_cosine = np.zeros(reward.shape[0])
        embedding_cosine_entity = np.zeros(reward.shape[0])

        for i in range(self.current_entities.shape[0]):

            if self.current_entities[i] < 10:
                continue

            cur = self.entity_id_to_cluster_mappping[str(self.current_entities[i])]
            current_cluster = self.mapping_dict[str(cur)]
            current_entity = self.current_entities[i]
            end_entity = self.end_entities[i]
            dot = np.dot(current_cluster, self.pretrained_entity_embeddings[current_entity].cpu())
            current_cluster_norm2 = np.linalg.norm(current_cluster)
            current_entity_norm2 = np.linalg.norm(self.pretrained_entity_embeddings[current_entity].cpu())
            end_entity_norm2 = np.linalg.norm(self.pretrained_entity_embeddings[end_entity].cpu())
            dot_2 = np.dot(self.pretrained_entity_embeddings[current_entity].cpu(), self.pretrained_entity_embeddings[end_entity].cpu())

            embedding_cosine[i] = np.divide(dot,np.multiply(current_cluster_norm2,current_entity_norm2))
            embedding_cosine_entity[i] = np.divide(dot_2,np.multiply(end_entity_norm2,current_entity_norm2))

        reward_2 = embedding_cosine

        lam = self.EGP.get_lam()

        reward = np.multiply(np.array((1 - np.array(lam))), np.array(reward_1)) + np.multiply(np.multiply(np.array(lam), np.array(oper)), np.array(reward_2))
        reward = reward.tolist()

        self.avg_cos_embb = embedding_cosine_entity.mean()
        self.EGP.update_lam(reward_1,embedding_cosine)
        return reward, reward_1

    def get_stepwise_approximated_reward(self, current_entities, current_clusters, prev_entities):

        credit = []
        num_rollout = int(current_entities.size(0) / self.batch_size)
        current_entities = current_entities.cpu().numpy()
        
        for i in range(0, len(current_entities), num_rollout):
            correct_num = 0.0
            num = 0.0
            for j in range(num_rollout):
                idx = i+j
                curr_ent = current_entities[idx]
                try:
                    ent2cls = self.entity_id_to_cluster_mappping[str(curr_ent)] # entity to cluster
                except:
                    continue
                curr_cls = current_clusters[idx]
                if curr_ent != 0:
                    num += 1.0
                    if ent2cls == curr_cls:
                        correct_num += 1.0
            if num == 0.0:
                credit.append(0.0)
            else:
                credit.append(correct_num/num)

        credit = torch.repeat_interleave(torch.tensor(credit), num_rollout)
        self.credits.append(credit)



    def __call__(self, action, prev_cluster, p_len):
        self.current_hop += 1
        self.current_entities = self.state['next_entities'][np.arange(self.no_examples*self.num_rollouts), action.cpu()]

        for i, cls in enumerate(prev_cluster): self.cluster_path[i].append(cls)

        next_actions, self.cluster_path, whether_e_agent_follows_c_agent, self.e_agent_cls_path = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                                                                  self.end_entities, self.all_entity_answers, self.current_hop == self.path_len - 1,
                                                                                                  self.num_rollouts, self.cluster_path, self.e_agent_cls_path, p_len)

        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities

        return self.state, whether_e_agent_follows_c_agent

    def get_avg_ent_emb(self):
        return self.avg_cos_embb


class ClusterEpisode(object):

    def __init__(self, graph, entity_data, cluster_data, params, mode):
        self.pretrained_entity_embeddings = params['pretrained_embeddings_entity']
        self.pretrained_relation_embeddings = params['pretrained_embeddings_relation']
        self.entity_id_to_cluster_mappping = params['entity_id_to_cluster_mappping']  # map
        self.grapher = graph
        self.avg_cos_emb = 0
        self.mode = mode
        if self.mode == 'train':
            self.num_rollouts = params['num_rollouts']
        else:
            self.num_rollouts = params['test_rollouts']

        self.batch_size = params['batch_size']
        self.positive_reward = params['positive_reward']
        self.negative_reward = params['negative_reward']
        self.alpha = params['alpha']

        self.path_len = params['path_length']
        self.test_rollouts = params['test_rollouts']
        self.mapping_dict = calc_cluster_embedding(self.pretrained_entity_embeddings, self.entity_id_to_cluster_mappping)

        self.current_hop = 0

        start_clusters, end_entities, all_answers = cluster_data
        self.no_examples = start_clusters.shape[0]  # original batch_size

        start_clusters = np.repeat(start_clusters, self.num_rollouts)
        end_clusters = np.repeat(end_entities, self.num_rollouts)  # [cls1, cls1, ..., cls2, cls2, ..., cls3, cls3, ...]

        self.start_clusters = start_clusters
        self.end_clusters = end_clusters
        self.current_clusters = start_clusters
        self.all_answers = all_answers

        next_actions = self.grapher.return_next_actions_cluster(self.current_clusters, self.start_clusters,
                                                                self.end_clusters, self.all_answers,
                                                                self.current_hop == self.path_len - 1,
                                                                self.num_rollouts)
        self.state = {}
        self.state['next_cluster_relations'] = next_actions[:, :, 1]  # shape: [original batch_size * num_rollout, max_num_actions]
        self.state['next_clusters'] = next_actions[:, :, 0]

        self.state['current_clusters'] = self.current_clusters


    def get_avg_clr_emb(self):
        return self.avg_cos_emb

    def get_reward(self):
        reward = (self.current_clusters == self.end_clusters)

        # set the True and False values to the values of positive and negative rewards.
        condlist = [reward == True, reward == False]
        choicelist = [self.positive_reward, self.negative_reward]
        # default_reward
        reward_1 = np.select(condlist, choicelist)  # [original batch_size * num_rollout]
        embeddings_cosine = np.zeros(reward.shape[0])
        for i in range(self.current_clusters.shape[0]):
            current_cluster = self.current_clusters[i]
            end_cluster = self.end_clusters[i]
            dot = np.dot(self.mapping_dict[str(current_cluster)], self.mapping_dict[str(end_cluster)])
            current_cluster_norm2 = np.linalg.norm(self.mapping_dict[str(current_cluster)]) # current cluster to embedding
            end_cluster_norm2= np.linalg.norm(self.mapping_dict[str(end_cluster)]) # end cluster to embedding
            embeddings_cosine[i] = np.divide(dot, np.multiply(current_cluster_norm2, end_cluster_norm2))

        self.avg_cos_emb = np.mean(embeddings_cosine)


        reward_2 = embeddings_cosine

        reward = reward_1 + self.alpha * reward_2
        return reward

    def get_query_cluster_relation(self):
        return self.end_clusters

    def get_state(self):
        return self.state

    def next_action(self, action):
        self.current_hop += 1
        self.current_clusters = self.state['next_clusters'][np.arange(self.no_examples * self.num_rollouts), action.cpu()]

        next_actions = self.grapher.return_next_actions_cluster(self.current_clusters, self.start_clusters,
                                                                self.end_clusters, self.all_answers,
                                                                self.current_hop == self.path_len - 1, self.num_rollouts)

        self.state['next_cluster_relations'] = next_actions[:, :, 1]  # shape: [original batch_size * num_rollout, max_num_actions]
        self.state['next_clusters'] = next_actions[:, :, 0]
        self.state['current_clusters'] = self.current_clusters
        return self.state



class env(object):
    def __init__(self, params, mode='train'):

        self.params = params
        self.mode = mode

        input_dir = params['data_input_dir']
        if mode == 'train':
            self.batcher = RelationEntityAndClusterBatcher(input_dir=input_dir,
                                                 batch_size=params['batch_size'],
                                                 entity_vocab=params['entity_vocab'],
                                                 relation_vocab=params['relation_vocab'],
                                                 cluster_vocab=params['cluster_vocab'],
                                                 cluster_relation_vocab=params['cluster_relation_vocab'],
                                                 entity_id_to_cluster_mappping=params['entity_id_to_cluster_mappping']
                                                 )
        else:
            self.batcher = RelationEntityAndClusterBatcher(input_dir=input_dir,
                                                           mode=mode,
                                                           batch_size=params['batch_size'],
                                                           entity_vocab=params['entity_vocab'],
                                                           relation_vocab=params['relation_vocab'],
                                                           cluster_vocab=params['cluster_vocab'],
                                                           cluster_relation_vocab=params['cluster_relation_vocab'],
                                                           entity_id_to_cluster_mappping=params['entity_id_to_cluster_mappping']
                                                           )

            self.total_no_examples = self.batcher.store.shape[0]

        self.entity_grapher = RelationEntityGrapher(triple_store=params['data_input_dir'] + '/' + 'graph.txt',
                                                    max_num_actions=params['max_num_actions'],
                                                    entity_vocab=params['entity_vocab'],
                                                    relation_vocab=params['relation_vocab'],
                                                    entity_id_to_cluster_mappping=params['entity_id_to_cluster_mappping'],
                                                    )

        self.cluster_grapher = RelationClusterGrapher(cluster_vocab=params['cluster_vocab'],
                                                      cluster_relation_vocab=params['cluster_relation_vocab']
                                                      )

    def get_episodes(self, batch_counter):

        if self.mode == 'train':
            for entity_data, cluster_data in self.batcher.yield_next_batch_train():
                yield EntityEpisode(self.entity_grapher, entity_data, cluster_data, self.params, self.mode), ClusterEpisode(self.cluster_grapher, entity_data,cluster_data, self.params, self.mode)
        else:
            for entity_data, cluster_data in self.batcher.yield_next_batch_test():
                if entity_data == None or cluster_data == None:
                    return
                yield EntityEpisode(self.entity_grapher, entity_data, cluster_data, self.params, self.mode), ClusterEpisode(self.cluster_grapher, entity_data,cluster_data, self.params, self.mode)
