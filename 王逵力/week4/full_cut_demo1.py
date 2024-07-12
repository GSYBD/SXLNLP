import os
import jieba


def read_and_process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    full_list = jieba.cut(text, cut_all=True)
    output_file_path = 'res.txt'
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write("/ ".join(full_list))
    print(f"分词结果已写入到 {output_file_path} 文件中")


if __name__ == "__main__":
    while True:
        input_file_path = input("请输入你要处理的文件路径：")
        if os.path.exists(input_file_path):
            try:
                read_and_process_file(input_file_path)
                break
            except Exception as e:
                print(f"处理文件时发生错误：{e}")
        else:
            print("没有找到指定的文件，请再次尝试。")