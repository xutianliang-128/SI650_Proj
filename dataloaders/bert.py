from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory

import torch
import torch.utils.data as data_utils


class BertDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        args.num_items = len(self.smap)
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.item_count + 1

        code = args.train_negative_sampler_code
        print(args.train_negative_sampling_seed)
        train_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                          self.user_count, self.item_count,
                                                          args.train_negative_sample_size,
                                                          args.train_negative_sampling_seed,
                                                          self.save_folder)
        code = args.test_negative_sampler_code
        test_negative_sampler = negative_sampler_factory(code, self.train, self.val, self.test,
                                                         self.user_count, self.item_count,
                                                         args.test_negative_sample_size,
                                                         args.test_negative_sampling_seed,
                                                         self.save_folder)

        self.train_negative_samples = train_negative_sampler.get_negative_samples()
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_dataset(self):
        dataset = BertTrainDataset(self.train, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.item_count,
                                   self.rng, self.train_time, self.train_r)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        dataset = self._get_eval_dataset(mode)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=False, pin_memory=True)
        return dataloader

    def _get_eval_dataset(self, mode):
        answers = self.val if mode == 'val' else self.test
        answers_t = self.val_time if mode == 'val' else self.test_time
        answers_r = self.val_r if mode == 'val' else self.test_r
        dataset = BertEvalDataset(self.train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples,
                                  self.train_time, self.train_r, answers_t, answers_r)
        return dataset


class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng, u2t, u2r):
        self.u2seq = u2seq
        self.u2t = u2t
        self.u2r = u2r
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self._getseq(user)

        tokens = []
        ratings = []
        times = []
        labels = []
        i = 0
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
                ratings.append(self.u2r[user][i])
            else:
                tokens.append(s)
                labels.append(0)

                ratings.append(self.u2r[user][i])

            times.append(self.u2t[user][i])

            i += 1

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        ratings = ratings[-self.max_len:]
        times = times[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        ratings = [0] * mask_len + ratings
        times = [0] * mask_len + times
        labels = [0] * mask_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels), torch.LongTensor(ratings), torch.LongTensor(times)

        #return torch.LongTensor(tokens), torch.LongTensor(labels)

    def _getseq(self, user):
        return self.u2seq[user]



class BertEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples, u2t, u2r, u2answer_t, u2answer_r):
        self.u2seq = u2seq
        self.u2t = u2t
        self.u2r = u2r
        self.u2answer_t = u2answer_t
        self.u2answer_r = u2answer_r
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]

        answer_t, answer_r = self.u2t[user] + self.u2answer_t[user], self.u2r[user] + self.u2answer_r[user]

        negs = self.negative_samples[user]

        candidates = answer + negs

        labels = [1] * len(answer) + [0] * len(negs)

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]

        answer_t = answer_t[-self.max_len:]
        answer_r = answer_r[-self.max_len:]

        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        answer_t = [0] * padding_len + answer_t
        answer_r = [0] * padding_len + answer_r

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels), \
               torch.LongTensor(answer_r), torch.LongTensor(answer_t)

