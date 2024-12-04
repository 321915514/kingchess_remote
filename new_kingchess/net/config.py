
CONFIG = {
    # 'kill_action': 30,      #和棋回合数
    'dirichlet': 0.2,       # 国际象棋，0.3；日本将棋，0.15；围棋，0.03
    'play_out': 1200, # test 10 run:1200        # 每次移动的模拟次数
    'c_puct': 5,             # u的权重
    'buffer_size': 10000,   # 经验池大小
    # 'paddle_model_path': 'current_policy.model',      # paddle模型路径
    'pytorch_model_path': './model/pytorch.pth',  # 初始参数 # pytorch模型路径
    "pytorch_current_model_path": './model/current.pth',  # 刚开始等于初始参数
    "pytorch_best_model_path": './model/best.pth',  # 刚开始等于初始参数
    "pytorch_current_onnx_path": './model/current.onnx',
    "pytorch_current_engine_path": './model/current.engine',  # 初始参数转换的引擎
    'train_data_buffer_path': './pickle/train_data_buffer.pkl',   # 数据容器的路径
    'batch_size': 512,  # 每次更新的train_step数量
    'kl_targ': 0.02,  # kl散度控制
    'epochs' : 5,  # 每次更新的train_step数量
    'game_batch_num': 3000,  # 训练更新的次数
    'use_frame': 'pytorch',  # paddle or pytorch根据自己的环境进行切换
    'train_update_interval': 30,  #模型更新间隔时间
    'use_redis': False, # 数据存储方式
    'redis_host': 'localhost',
    'redis_port': 6379,
    'redis_db': 0,
}


CONFIG_LDCONV = {
    # 'kill_action': 30,      #和棋回合数
    'dirichlet': 0.2,       # 国际象棋，0.3；日本将棋，0.15；围棋，0.03
    'play_out': 1200, # test 10 run:1200        # 每次移动的模拟次数
    'c_puct': 5,             # u的权重
    'buffer_size': 10000,   # 经验池大小
    # 'paddle_model_path': 'current_policy.model',      # paddle模型路径
    'pytorch_model_path': './ldconv_model/pytorch.pth',  # 初始参数 # pytorch模型路径
    "pytorch_current_model_path": './ldconv_model/current.pth',  # 刚开始等于初始参数
    "pytorch_best_model_path": './ldconv_model/best.pth',  # 刚开始等于初始参数
    "pytorch_current_onnx_path": './ldconv_model/current.onnx',
    "pytorch_current_engine_path": './ldconv_model/current.engine',  # 初始参数转换的引擎
    'train_data_buffer_path': './pickle/train_data_buffer.pkl',   # 数据容器的路径
    'batch_size': 512,  # 每次更新的train_step数量
    'kl_targ': 0.02,  # kl散度控制
    'epochs' : 5,  # 每次更新的train_step数量
    'game_batch_num': 3000,  # 训练更新的次数
    'use_frame': 'pytorch',  # paddle or pytorch根据自己的环境进行切换
    'train_update_interval': 30,  #模型更新间隔时间
    'use_redis': False, # 数据存储方式
    'redis_host': 'localhost',
    'redis_port': 6379,
    'redis_db': 0,
}


CONFIG_SUPERVISE = {
    # 'kill_action': 30,      #和棋回合数
    'dirichlet': 0.2,       # 国际象棋，0.3；日本将棋，0.15；围棋，0.03
    'play_out': 1200, # test 10 run:1200        # 每次移动的模拟次数
    'c_puct': 5,             # u的权重
    'buffer_size': 10000,   # 经验池大小
    # 'paddle_model_path': 'current_policy.model',      # paddle模型路径
    'pytorch_model_path': './supervise_model/current_best_white_0.36.pth',  # 初始参数 # pytorch模型路径
    "pytorch_current_model_path": './supervise_model/current.pth',  # 刚开始等于初始参数
    "pytorch_best_model_path": './supervise_model/best.pth',  # 刚开始等于初始参数
    "pytorch_current_onnx_path": './supervise_model/current.onnx',
    "pytorch_current_engine_path": './supervise_model/current.engine',  # 初始参数转换的引擎
    'train_data_buffer_path': './pickle/train_data_buffer.pkl',   # 数据容器的路径
    'batch_size': 512,  # 每次更新的train_step数量
    'kl_targ': 0.02,  # kl散度控制
    'epochs' : 5,  # 每次更新的train_step数量
    'game_batch_num': 3000,  # 训练更新的次数
    'use_frame': 'pytorch',  # paddle or pytorch根据自己的环境进行切换
    'train_update_interval': 30,  #模型更新间隔时间
    'use_redis': False, # 数据存储方式
    'redis_host': 'localhost',
    'redis_port': 6379,
    'redis_db': 0,
}


CONFIG_TRAIN= {
    # 'kill_action': 30,      #和棋回合数
    'dirichlet': 0.2,  # 国际象棋，0.3；日本将棋，0.15；围棋，0.03
    'play_out': 1200,  # test 10 run:1200        # 每次移动的模拟次数
    'c_puct': 5,  # u的权重
    'buffer_size': 100000,  # 经验池大小
    # 'paddle_model_path': 'current_policy.model',      # paddle模型路径
    'pytorch_model_path': './model_torch/pytorch.pth',  # 初始参数 # pytorch模型路径
    "pytorch_current_model_path": './model_torch/current.pth',  # 刚开始等于初始参数
    "pytorch_best_model_path": './model_torch/best.pth',  # 刚开始等于初始参数
    "pytorch_current_onnx_path": './model_torch/current.onnx',
    "pytorch_current_engine_path": './model_torch/current.engine',  # 初始参数转换的引擎
    'train_data_buffer_path': './pickle/train_data_buffer.pkl',  # 数据容器的路径
    'batch_size': 512,  # 每次更新的train_step数量
    'kl_targ': 0.02,  # kl散度控制
    'epochs': 5,  # 每次更新的train_step数量
    'game_batch_num': 3000,  # 训练更新的次数
    'use_frame': 'pytorch',  # paddle or pytorch根据自己的环境进行切换
    'train_update_interval': 30,  # 模型更新间隔时间
    'use_redis': False,  # 数据存储方式
    'redis_host': 'localhost',
    'redis_port': 6379,
    'redis_db': 0,
}
CONFIG_TRANSFORMER= {
    # 'kill_action': 30,      #和棋回合数
    'dirichlet': 0.2,  # 国际象棋，0.3；日本将棋，0.15；围棋，0.03
    'play_out': 1200,  # test 10 run:1200        # 每次移动的模拟次数
    'c_puct': 5,  # u的权重
    'buffer_size': 100000,  # 经验池大小
    # 'paddle_model_path': 'current_policy.model',      # paddle模型路径
    'pytorch_model_path': './model_transformer/pytorch.pth',  # 初始参数 # pytorch模型路径
    "pytorch_current_model_path": './model_transformer/current.pth',  # 刚开始等于初始参数
    "pytorch_best_model_path": './model_transformer/best.pth',  # 刚开始等于初始参数
    "pytorch_current_onnx_path": './model_transformer/current.onnx',
    "pytorch_current_engine_path": './model_transformer/current.engine',  # 初始参数转换的引擎
    'train_data_buffer_path': './pickle/train_data_buffer.pkl',  # 数据容器的路径
    'batch_size': 512,  # 每次更新的train_step数量
    'kl_targ': 0.02,  # kl散度控制
    'epochs': 5,  # 每次更新的train_step数量
    'game_batch_num': 3000,  # 训练更新的次数
    'use_frame': 'pytorch',  # paddle or pytorch根据自己的环境进行切换
    'train_update_interval': 30,  # 模型更新间隔时间
    'use_redis': False,  # 数据存储方式
    'redis_host': 'localhost',
    'redis_port': 6379,
    'redis_db': 0,
}


CONFIG_LINEAR= {
    # 'kill_action': 30,      #和棋回合数
    'dirichlet': 0.2,  # 国际象棋，0.3；日本将棋，0.15；围棋，0.03
    'play_out': 1200,  # test 10 run:1200        # 每次移动的模拟次数
    'c_puct': 5,  # u的权重
    'buffer_size': 100000,  # 经验池大小
    # 'paddle_model_path': 'current_policy.model',      # paddle模型路径
    'pytorch_model_path': './model_linear/pytorch.pth',  # 初始参数 # pytorch模型路径
    "pytorch_current_model_path": './model_linear/current.pth',  # 刚开始等于初始参数
    "pytorch_best_model_path": './model_linear/best.pth',  # 刚开始等于初始参数
    "pytorch_current_onnx_path": './model_linear/current.onnx',
    "pytorch_current_engine_path": './model_linear/current.engine',  # 初始参数转换的引擎
    'train_data_buffer_path': './pickle/train_data_buffer.pkl',  # 数据容器的路径
    'batch_size': 512,  # 每次更新的train_step数量
    'kl_targ': 0.02,  # kl散度控制
    'epochs': 5,  # 每次更新的train_step数量
    'game_batch_num': 3000,  # 训练更新的次数
    'use_frame': 'pytorch',  # paddle or pytorch根据自己的环境进行切换
    'train_update_interval': 30,  # 模型更新间隔时间
    'use_redis': False,  # 数据存储方式
    'redis_host': 'localhost',
    'redis_port': 6379,
    'redis_db': 0,
}
