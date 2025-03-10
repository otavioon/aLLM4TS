from data_provider.data_loader import (
    Dataset_ETT_hour, 
    Dataset_ETT_minute, 
    Dataset_Custom,
    Dataset_Weather,
    Dataset_Traffic,
    Dataset_Electricity, 
    Dataset_pretrain,
    PSMSegLoader,
    MSLSegLoader, 
    SMAPSegLoader, 
    SMDSegLoader, 
    SWATSegLoader,
    Dataset_M4,
    UEAloader,
    DAGHAR,
    Pretrain_allm4ts_ES_dataset,
    Pretrain_allm4ts_DAGHAR_dataset
    
)
from torch.utils.data import DataLoader
import torch
from data_provider.uea import collate_fn

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'weather': Dataset_Weather,
    'traffic': Dataset_Traffic,
    "electricity": Dataset_Electricity,
    'custom': Dataset_Custom,
    'pretrain': Dataset_pretrain,
    "pretrain_allm4ts_es": Pretrain_allm4ts_ES_dataset,
    "pretrain_allm4ts_daghar": Pretrain_allm4ts_DAGHAR_dataset,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWaT': SWATSegLoader,
    'm4': Dataset_M4,
    'UEA': UEAloader,
    'DAGHAR': DAGHAR,
}



def data_provider(args, flag):
    Data = data_dict[args.data]     # Dataset_Pretrain for 'pretrain
    timeenc = 0 if args.embed != 'timeF' else 1 # timeF for default in pretrain (1)

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq                # defaults to H in pretrain
    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
    elif args.task_name =='short_term_forecast':
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        batch_size = args.batch_size
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            # shuffle=False,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        print("**************************Data loader length: ", len(data_loader))
        
        return data_set, data_loader
    else:       # Here we are in the pretrain task
        data_set = Data(
            configs=args,
            root_path=args.root_path,       # "./dataset/"
            data_path=args.data_path,       # null for pretrain
            flag=flag,                      # train, test, val
            size=[args.seq_len, args.label_len, args.pred_len], # 1024, 0, 1024 for pretrain
            features=args.features,         # M  for pretrain
            target=args.target,             # OT
            timeenc=timeenc,                # 1 for pretrain
            freq=freq,                      # defaults to h in pretrain
            percent=args.percent            # 100 for pretrain
        )

    if args.use_multi_gpu and args.use_gpu and flag == 'train':
        if flag == 'train':
            train_sampler = torch.utils.data.distributed.DistributedSampler(data_set)
            data_loader = DataLoader(data_set, batch_size=batch_size, sampler=train_sampler, num_workers=args.num_workers, drop_last=drop_last)
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
    return data_set, data_loader

