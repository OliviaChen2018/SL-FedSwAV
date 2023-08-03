from nvidia.dali.plugin.pytorch import DALIGenericIterator
import pdb

class DALIDataloader(DALIGenericIterator):
    # 传参时size==size = len(sampler) / num_shards
    def __init__(self, sampler, pipeline, batch_size, size, output_map=["data", "label"], auto_reset=True, onehot_label=False, shuffle = True):
#         super(DALIDataloader, self).__init__(pipelines=pipeline, size=size, auto_reset=auto_reset, output_map=output_map)
        super(DALIDataloader, self).__init__(pipelines=pipeline, size=size,output_map=output_map)
#         self.size = size #这一步赋值不能执行,不知道为什么...
        self.batch_size = batch_size
        self.onehot_label = onehot_label
        self.output_map = output_map #output_map是指定batch中的字典的两个key
        self.sampler = sampler
        
    def __next__(self):
#         pdb.set_trace()
        if self._first_batch is not None: # 只有第一个batch走这里
            batch = self._first_batch
            self._first_batch = None
#             pdb.set_trace()
            return batch[0]  #batch是一个list, list中有一个dict.
        # batch==[{'data': tensor([batch_size, channel_size, input_size, input_size]), 
        #               'labels': tensor([batch_size, 1])}]
        
        data = super().__next__()[0]
#         pdb.set_trace()
        if self.onehot_label:
            data[self.output_map[1]] = data[self.output_map[1]].squeeze().long()
#             return [data[self.output_map[0]], data[self.output_map[1]].squeeze().long()]
#         else:
#             return [data[self.output_map[0]], data[self.output_map[1]]]
#         if self._ever_consumed is True:
#             pdb.set_trace()
        return data
    
    def __len__(self): #计算batch的数量
        if self._size % self.batch_size==0: 
         #self._size是调用DALIGenericIterator类的属性,该值的大小等于训练集的样本数量
            return self._size // self.batch_size
        else:
            return self._size // self.batch_size+1
        
    def reset(self, epoch, num_shards=1, shard_id=0, shuffle = True):
        self.sampler.reset_data(epoch, num_shards, shard_id, shuffle = True)
        for p in self._pipes: # 在源码中self._pipes = pipelines
            p.reset_iterator(self.sampler)
#         pdb.set_trace()
        super(DALIDataloader, self).reset()
        
        
class DALIDataloader_DDP(DALIGenericIterator):
    # 传参时size==size = len(sampler) / num_shards
    def __init__(self, sampler, pipeline, batch_size, size, output_map=["data", "label"], auto_reset=True, onehot_label=False, shuffle = True, is_distributed = False):
        super(DALIDataloader, self).__init__(pipelines=pipeline, size=size, auto_reset=auto_reset, output_map=output_map)
#         self.size = size #这一步赋值不能执行,不知道为什么...
        self.batch_size = batch_size
        self.onehot_label = onehot_label
        self.output_map = output_map #output_map是指定batch中的字典的两个key
        self.sampler = sampler
        
    def __next__(self):
#         pdb.set_trace()
        if self._first_batch is not None: # 只有第一个batch走这里
            batch = self._first_batch
            self._first_batch = None
#             pdb.set_trace()
            return batch[0]  #batch是一个list, list中有一个dict.
        # batch==[{'data': tensor([batch_size, channel_size, input_size, input_size]), 
        #               'labels': tensor([batch_size, 1])}]
        data = super().__next__()[0]
        
        if self.onehot_label:
            data[self.output_map[1]] = data[self.output_map[1]].squeeze().long()
#             return [data[self.output_map[0]], data[self.output_map[1]].squeeze().long()]
#         else:
#             return [data[self.output_map[0]], data[self.output_map[1]]]
        return data
    
    def __len__(self): #计算batch的数量
        if self._size % self.batch_size==0: 
         #self._size是调用DALIGenericIterator类的属性,该值的大小等于训练集的样本数量
            return self._size // self.batch_size
        else:
            return self._size // self.batch_size+1
        
    def reset(self, epoch, num_shards=1, shard_id=0, shuffle = True):
        self.sampler.reset_data(epoch, num_shards, shard_id, shuffle = True)
        for p in self._pipes: # 在源码中self._pipes = pipelines
            p.reset_iterator(self.sampler)
        super(DALIDataloader, self).reset()