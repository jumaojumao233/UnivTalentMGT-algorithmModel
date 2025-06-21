model_list=[
    'decission_tree',
]
import os

# 检查文件是否存在

for modelName in model_list:
    file_exists = os.path.exists('saved_model/'+modelName+".pkl")
    if file_exists:
        pass
    else:
        print(modelName)
