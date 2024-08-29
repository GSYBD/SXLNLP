"""
isinstance
https://blog.csdn.net/weixin_65190179/article/details/138498641
"""

def process_data(data):
    if isinstance(data, list):
        print("处理列表数据")
    elif isinstance(data, dict):
        print("处理字典数据")
    elif isinstance(data, str):
        print("处理字符串数据")
    else:
        print("未知数据类型")
