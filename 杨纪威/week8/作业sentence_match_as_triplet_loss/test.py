"""
isinstance
https://blog.csdn.net/weixin_65190179/article/details/138498641
"""

def process_data(data):
    if isinstance(data, list):
        print("�����б�����")
    elif isinstance(data, dict):
        print("�����ֵ�����")
    elif isinstance(data, str):
        print("�����ַ�������")
    else:
        print("δ֪��������")
