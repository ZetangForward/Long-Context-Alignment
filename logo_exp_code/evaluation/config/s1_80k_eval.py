class EvalConfig:
    model_path = "/nvme/zecheng/ckpt/v3"
    peft_path = None
    rope_theta = None
    rope_factor = None
    rope_type = None
    max_position_embeddings = 65536
    rope_theta = 200e6 
    model_max_length_setting = "long_setting"
    save_path = "./longbench"
    model_name = f"test-${model_max_length_setting}-logn_scaling"
    max_training_length = 16384