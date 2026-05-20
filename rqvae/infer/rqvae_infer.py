import collections
import json
import logging
import argparse
import torch.distributed as dist
import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import CustomNpzFile
from rqvae_model import RQVAE
import os
import glob

def select_files(part, total_parts=20, file_count=20):
    """
    根据part参数选择对应的文件范围

    参数:
        part: 要选择的部分编号(0-9)
        total_parts: 总部分数(默认为10)
        file_count: 总文件数(默认为100，即part-00000到part-00099)

    返回:
        所选部分的文件名列表
    """
    if part < 0 or part >= total_parts:
        raise ValueError(f"part参数必须在0到{total_parts - 1}之间")

    files_per_part = file_count // total_parts
    start = part * files_per_part
    end = start + files_per_part - 1

    # 处理最后一个part可能包含多余文件的情况
    if part == total_parts - 1:
        end = file_count - 1

    selected_files = []
    for i in range(start, end + 1):
        # 格式化数字为5位，前面补零
        file_num = i
        # selected_files.append(f"worker_{file_num}.npz")
        selected_files.append(f"worker_{file_num}.npz")

    return selected_files

def parse_args_from_json(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    config_dict={}
    for keys,value_dict in config.items():
        config_dict[keys] = {}
        for k,v in value_dict.items():
            config_dict[keys][k] = v
    print(config_dict)

    return config_dict
def my_collate(data):
    item_list, image_list = [], []
    for item in data:
        item, emb = item
        if type(emb) != int :
            item_list.append(item)
            image_list.append(emb)
    return item_list, image_list
def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice


def check_collision_dict(all_indices_dict):
    print('In check_collision_dict')
    # 获取字典中所有值的列表
    all_indices_str = list(all_indices_dict.values())

    # 计算总项数
    tot_item = len(all_indices_str)

    # 计算唯一项数
    tot_indice = len(set(all_indices_str))

    # 返回是否所有项都是唯一的
    print('Return check_collision_dict')
    return tot_item == tot_indice


def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups


def get_collision_item_dict(all_indices_dict):
    index2id = {}
    for key, value in all_indices_dict.items():
        if value not in index2id:
            index2id[value] = []
        index2id[value].append(key)

    collision_item_groups = []

    for value in index2id:
        if len(index2id[value]) > 1:
            collision_item_groups.append(index2id[value])

    return collision_item_groups

def custom_collate_fn(batch):
    # 处理 NumPy 标量
    batch = [
        torch.as_tensor(x) if isinstance(x, (np.ndarray, np.generic)) else x
        for x in batch
    ]
    return torch.utils.data.default_collate(batch)

def normalize_item_id(item_id):
    if isinstance(item_id, torch.Tensor):
        return int(item_id.detach().cpu().reshape(-1)[0].item())
    if isinstance(item_id, np.ndarray):
        return int(item_id.reshape(-1)[0])
    if isinstance(item_id, np.generic):
        return int(item_id.item())
    return int(item_id)
    
def main(worker_path_list, output_dir):
    # ckpt_path = os.environ.get("USER_CACHE_PATH") + "/emb_test/2048_2048_1024/best_collision_model.pth"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ckpt_candidates = glob.glob(os.path.join(project_root, "outputs", "rqvae", "ckpt", "*", "best_collision_model.pth"))
    if not ckpt_candidates:
        raise FileNotFoundError("No best_collision_model.pth found under outputs/rqvae/ckpt/*/")
    ckpt_path = max(ckpt_candidates, key=os.path.getmtime)
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'),weights_only=False)
    args = ckpt["args"]
    state_dict = ckpt["state_dict"]
    """build dataset"""
    data = CustomNpzFile(worker_path_list)
    print(len(data))
    data_loader = DataLoader(data, num_workers=4,batch_size=81920)
    """build model"""
    model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  )
    model.load_state_dict(state_dict)
    model=model.cuda()
    model.eval()
    # dp_model = torch.nn.DataParallel(model)
    # dp_model.eval()
    # print(dp_model)

    all_indices = {}
    all_indices_str = {}
    item2global_dict = {}
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>"]
    global_index = 0
    print(len(data_loader))
    for d_idx, d in enumerate(tqdm(data_loader)):
        torch.cuda.empty_cache()
        item_ids, tensor_embs = d[0],d[1]
        item_ids = list(item_ids)# 转换为列表
        # tensor_embs = torch.stack(tensor_embs).cuda() # 转换为张量并移动到设备
        tensor_embs = tensor_embs.cuda()  # 转换为张量并移动到设备
        # d = d.to(device)
        try:
            indices = model.get_indices(tensor_embs, use_sk=False)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item_index, index in enumerate(indices):
                code = []
                for i, ind in enumerate(index):
                    code.append(prefix[i].format(int(ind)))
                item_id = normalize_item_id(item_ids[item_index])
                all_indices[item_id] = code
                all_indices_str[item_id] = str(code)
                item2global_dict[item_id] = global_index
                global_index += 1
        except:
            continue

    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    # model.rq.vq_layers[-1].sk_epsilon = 0.005
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    tt = 0
    # set infer times
    max_tt =  50
    # 设置冲突处理的batch size，可以根据显存大小调整
    collision_batch_size = 8192
    
    # There are often duplicate items in the dataset, and we no longer differentiate them
    print('Before while loop')
    while True:
        print('In while loop')
        # if tt >= 1 or check_collision_dict(all_indices_str):
        if tt >= max_tt or check_collision_dict(all_indices_str):
            break
        print("Before get_collision_item_dict:")
        collision_item_groups = get_collision_item_dict(all_indices_str)
        # print(collision_item_groups)
        print("The collision item groups length is:")
        print(len(collision_item_groups))

        # 收集所有冲突的items进行batch处理
        all_collision_items = []
        collision_item_to_group_map = {}  # 用于记录每个item属于哪个冲突组
        
        for group_idx, collision_items in enumerate(collision_item_groups):
            for item in collision_items:
                all_collision_items.append(item)
                collision_item_to_group_map[item] = group_idx
        
        print(f"Total collision items to process: {len(all_collision_items)}")
        
        # 按batch_size分批处理所有冲突items
        for batch_start in tqdm(range(0, len(all_collision_items), collision_batch_size), 
                               desc="Processing collision batches"):
            batch_end = min(batch_start + collision_batch_size, len(all_collision_items))
            batch_collision_items = all_collision_items[batch_start:batch_end]
            
            # 获取batch中所有items的indices
            batch_collision_indices = [item2global_dict[normalize_item_id(key)] for key in batch_collision_items]
            
            try:
                # 批量获取数据
                batch_data = data[batch_collision_indices]
                batch_item_ids, batch_tensor_embs = batch_data
                batch_item_ids = list(batch_item_ids)  # 转换为列表
                batch_tensor_embs = batch_tensor_embs.cuda()  # 转换为张量并移动到设备
                
                # 批量推理
                batch_indices = model.get_indices(batch_tensor_embs, use_sk=True)
                batch_indices = batch_indices.view(-1, batch_indices.shape[-1]).cpu().numpy()
                
                # 更新结果
                for collision_item, index, item_id in zip(batch_collision_items, batch_indices, batch_item_ids):
                    code = []
                    for i, ind in enumerate(index):
                        code.append(prefix[i].format(int(ind)))
                    item = normalize_item_id(item_id)
                    assert normalize_item_id(collision_item) == item
                    all_indices[item] = code
                    all_indices_str[item] = str(code)
                    
            except Exception as e:
                print(f"Error processing batch {batch_start}-{batch_end}: {e}")
                # 如果batch处理失败，回退到逐个处理
                print("Falling back to individual processing for this batch...")
                for item in batch_collision_items:
                    try:
                        item = normalize_item_id(item)
                        collision_item_index = item2global_dict[item]
                        d = data[collision_item_index]
                        item_id, tensor_emb = d
                        if isinstance(tensor_emb, (list, tuple)):
                            tensor_emb = torch.stack(tensor_emb)
                        tensor_emb = tensor_emb.unsqueeze(0).cuda()  # 添加batch维度
                        
                        indices = model.get_indices(tensor_emb, use_sk=True)
                        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
                        
                        code = []
                        for i, ind in enumerate(indices[0]):
                            code.append(prefix[i].format(int(ind)))
                        
                        all_indices[item] = code
                        all_indices_str[item] = str(code)
                    except Exception as inner_e:
                        print(f"Error processing individual item {item}: {inner_e}")
                        continue
            
            # 清理显存
            torch.cuda.empty_cache()
        
        tt += 1

    print("All indices number: ", len(all_indices))
    conflict_counts = get_indices_count(all_indices_str.values())
    max_conflicts = max(conflict_counts.values()) if conflict_counts else 0
    print(f"Max number of conflicts (same code shared by items): {max_conflicts}")

    tot_item = len(all_indices_str.values())
    tot_indice = len(set(list(all_indices_str.values())))
    print("Total number of items: ", tot_item)
    print("Total number of indices: ", tot_indice)
    print("Collision Rate", (tot_item - tot_indice) / tot_item)
    used_code_sets = [set() for _ in range(len(args.num_emb_list))]
    for code_seq in all_indices.values():
        for layer_idx, token in enumerate(code_seq[:len(used_code_sets)]):
            code_idx = int(token.rsplit("_", 1)[1].rstrip(">"))
            used_code_sets[layer_idx].add(code_idx)

    combo_total = 1
    for num_emb in args.num_emb_list[:len(used_code_sets)]:
        combo_total *= num_emb

    for layer_idx, (used_codes, num_emb) in enumerate(zip(used_code_sets, args.num_emb_list)):
        utilization = len(used_codes) / num_emb
        print(
            f"Layer {layer_idx} codebook utilization: "
            f"{len(used_codes)}/{num_emb} ({utilization:.6%})"
        )
    print(
        f"Combination utilization: {tot_indice}/{combo_total} "
        f"({tot_indice / combo_total:.6%})"
    )

    # all_indices_dict = {}
    # for item, indices in enumerate(all_indices.tolist()):
    #     all_indices_dict[item] = list(indices)

    output_file = os.path.join(output_dir,'worker_0_output.txt')
    with open(output_file, 'w') as fp:
        for item_id, code in all_indices.items():
            int_value_item = int(item_id)
            token = ''.join(code)
            fp.write('{}\t{}\n'.format(int_value_item, token))



if __name__ == '__main__':
    # Init some configs

    # data_path = "/emb/emb/"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_path = os.path.join(project_root, "outputs", "sasrec", "cache", "emb", "emb", "emb")
    data_path_list = [os.path.join(data_path, i)
                      for i in os.listdir(data_path) if '.npz' in i]

    # output_dir = os.path.join(os.environ.get("USER_CACHE_PATH"), "emb_infer/sinkhorn50")
    output_dir = os.path.join(project_root, "outputs", "rqvae", "emb_infer", "sinkhorn50")

    # 确保目录存在（如果不存在则创建）
    os.makedirs(output_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错


    main(data_path_list, output_dir)
