def full_segmentation(text, dictionary):
    # 初始化结果列表
    results = []

    # 递归函数实现全切分
    def segment(text, current):
        if not text:
            results.append(current)
            return
        for i in range(1, len(text) + 1):
            word = text[:i]
            if word in dictionary:
                segment(text[i:], current + [word])

    # 开始全切分
    segment(text, [])
    return results

# 示例词典
dictionary = {"我", "喜欢", "北京", "天安门", "天", "安", "门", "喜欢北京", "北京天安门"}

# 示例文本
text = "我喜欢北京天安门"

# 获取全切分结果
segments = full_segmentation(text, dictionary)

# 打印结果
for seg in segments:
    print(" ".join(seg))