import torch as t

from nnsight import LanguageModel


class ActivationBuffer():
    def __init__(self, model, dataset, activation_dim, batch_size=512, buffer_size=8):
        self.model = model
        self.batch_size = batch_size
        self.buffer_size = batch_size * buffer_size
        self.activation_dim = activation_dim
        self.dataset = dataset

        self.refresh()

    def __iter__(self):
        return self
    
    def __next__(self):
        print(self.n_acts)
        if self.buf_pos + self.batch_size >= self.n_acts:
            self.refresh()
        
        batch = self.acts[self.buf_pos:self.buf_pos+self.batch_size]
        self.buf_pos += self.batch_size
        return batch
    
    def refresh(self):
        self.n_acts = 0
        self.buf_pos = 0

        hidden_list = []
        while self.n_acts < self.buffer_size:
            str_batch = next(self.dataset)['text']

            """
            token_batch = self.model.tokenizer(
                str_batch,
                return_tensors='pt',
    #            max_length=,
                padding=True,
                truncation=True
            )

            tokens = token_batch['input_ids']
            """


            with t.no_grad(), self.model.trace(str_batch) as tracer:
                hidden = self.model.gpt_neox.layers[0].output[0].save()
            
            print(hidden.shape)
            hidden = hidden.flatten(start_dim=0, end_dim=-2)
            hidden_list.append(hidden)
            
            self.n_acts += hidden.size(0)
        

        self.acts = t.cat(hidden_list, dim=0)
        self.acts = self.acts[t.randperm(self.acts.size(0))]
