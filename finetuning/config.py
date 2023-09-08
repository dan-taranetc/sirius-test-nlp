# Data Params
data_dir_path = './data_output'
val_size = 0.2

# Train params
num_train_epochs=1,             
per_device_train_batch_size=8,  
per_device_eval_batch_size=8,   
warmup_steps=500,               
weight_decay=0.01,              
fp16=True                       

# HF configs
model_source = "taranetsdan/ruDialoGPT_v2_medium"
hf_token = ""