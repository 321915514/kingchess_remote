# import concurrent
import os
import random

import numpy as np
from torch.utils.data import Dataset
import train_pb2
from fundamental.utils import print_board
from net.encoder import decoder_board
import threading
import concurrent.futures

class DecodeDataset:
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.file_paths = []
        # self.read_dir()

    def read_dir(self):
        # if os.path.isdir(self.dataset):
        self.file_paths = [os.path.join(self.dataset, file) for file in os.listdir(self.dataset)]


    # def get_data(self):
    #     self.read_dir()
    #     for file_path in self.file_paths:
    #         try:
    #             train_list = self.parse_file(file_path)
    #             yield train_list
    #             # 删除已经处理过的文件
    #             os.remove(file_path)
    #         except Exception as e:
    #             # 处理异常，例如文件不存在或处理文件时出错
    #             print(f"Error processing file {file_path}: {e}")
    #             continue
    #             # 根据需要决定是否继续迭代或中断生成器
    #             # 如果选择中断，可以抛出异常或使用其他机制

    def process_files_concurrently(self):
        self.read_dir()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:  # 设置线程池大小
            #random.shuffle(self.file_paths)
            #select_file = random.sample(self.file_paths, 10)
            future_to_file = {executor.submit(self.parse_file, file_path): file_path for file_path in self.file_paths}
            result = []
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    train_list = future.result()  # 获取处理结果
                    # 在这里可以使用train_list，比如将其加入一个列表或进行其他操作
                    # print(f"Processed {file_path}")

                    # 删除已经处理过的文件（请小心操作，确保不会误删重要文件）
                    if os.path.exists(file_path):
                        os.remove(file_path)

                    result.extend(train_list)
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue
            return result

    def parse_file(self, file_path):
        game_file = open(file_path, 'rb').read()

        train_data_array = train_pb2.TripletArray()

        train_data_array.ParseFromString(game_file)

        train_list = []

        for triplet in train_data_array.triplets:
            array_list = []
            move = [] # 2024/4/4 modify
            for arr in triplet.array:
                array_list.append(arr)
            # print(len(array_list))
            # board_np = np.array(array_list).reshape((5, 9, 9))
            board_np = np.array(array_list).reshape((32, 5, 9))


           # print(board_np)
            

            # transformer
            # board_np = np.array(array_list[:243])
            # board_real, player = decoder_board(board_np)
            # print_board(board_real)
            for d in triplet.dictionary:
                move.append(d)
            moves = np.array(move, dtype=float)
            value = triplet.value
            #print(value)
            train_list.append((board_np, moves, value))

        return train_list


if __name__ == '__main__':
    # read_file('abf5bb8e-d508-4eeb-9427-cd57258f7862.game')
    decodeDataset = DecodeDataset('./dataset')
    generator = decodeDataset.process_files_concurrently()  # 一个generator就是一个game文件

    # print(generator)

    for item in generator:
        print(len(item))

    # a = next(generator)
    #
    # print(len(a))
