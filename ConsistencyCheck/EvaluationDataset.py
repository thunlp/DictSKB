from consistency_utils import *
import random
import torch

dict_all_senses_sememes, hownet_all_senses_sememes = get_all_sensesAndsememes()
dict_all_senses_sememes, hownet_all_senses_sememes = dict_all_senses_sememes[:100000], hownet_all_senses_sememes[
                                                                                       :100000]
dict_sememe_list, hownet_sememe_list = get_sememe_list()

def getEvaluationDictdata(dict_sememes_embedding, device, ):
    with torch.no_grad():
        # random.shuffle(dict_all_senses_sememes)
        processed_all_senses_sememes_vec = []
        all_senses_sememes_tag_tensors = torch.zeros(100000, len(dict_sememe_list))  # number of senses, sememes number

        for i, sememe_set in enumerate(dict_all_senses_sememes):
            inputs = torch.tensor(list(sememe_set)).long().to(device)
            sememes_embeddings = dict_sememes_embedding(inputs) # shape: number of sememes, 200
            sense_vec = torch.sum(sememes_embeddings, dim=0) / float(len(sememe_set))
            processed_all_senses_sememes_vec.append(sense_vec)
            for sememe_id in sememe_set:
                all_senses_sememes_tag_tensors[i, sememe_id] = 1

        refer_data_vec = torch.stack(processed_all_senses_sememes_vec, dim=0)
        # refer_data_sememes_tag = all_senses_sememes_tag_tensors[0:90000, :]

        # test_data_vec = torch.stack(processed_all_senses_sememes_vec[90000:], dim=0)
        # test_data_sememes_tag = all_senses_sememes_tag_tensors[90000:, :]
        # return refer_data_vec, refer_data_sememes_tag, test_data_vec, test_data_sememes_tag
        return refer_data_vec, all_senses_sememes_tag_tensors





def getEvaluationHownetdata(hownet_sememes_embedding, device):
    with torch.no_grad():
        random.shuffle(hownet_all_senses_sememes)
        processed_all_senses_sememes_vec = []
        all_senses_sememes_tag_tensors = torch.zeros(100000, len(hownet_sememe_list))  # number of senses, sememes number

        for i, sememe_set in enumerate(hownet_all_senses_sememes):
            inputs = torch.tensor(list(sememe_set)).long().to(device)
            sememes_embeddings = hownet_sememes_embedding(inputs) # shape: number of sememes, 200
            sense_vec = torch.sum(sememes_embeddings, dim=0) / float(len(sememe_set))
            processed_all_senses_sememes_vec.append(sense_vec)
            for sememe_id in sememe_set:
                all_senses_sememes_tag_tensors[i, sememe_id] = 1

        refer_data_vec = torch.stack(processed_all_senses_sememes_vec, dim=0)
        # refer_data_sememes_tag = all_senses_sememes_tag_tensors[0:90000, :]


        # test_data_vec = torch.stack(processed_all_senses_sememes_vec[90000:], dim=0)
        #
        # test_data_sememes_tag = all_senses_sememes_tag_tensors[90000:, :]
        return refer_data_vec, all_senses_sememes_tag_tensors
