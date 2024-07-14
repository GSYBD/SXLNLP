# 定义一个词典，其中包含词语及其出现的频率
Dict = {
    "经常": 0.1,
    "经": 0.05,
    "有": 0.1,
    "常": 0.001,
    "有意见": 0.1,
    "歧": 0.001,
    "意见": 0.2,
    "分歧": 0.2,
    "见": 0.05,
    "意": 0.05,
    "见分歧": 0.05,
    "分": 0.1
}

# 定义 Trie 树的节点
class TrieNode:
    def __init__(self):
        self.children = {}  # 存储子节点
        self.is_end_of_word = False  # 标记这个节点是否是某个单词的结尾

# Trie 树的实现
class Trie:
    def __init__(self):
        self.root = TrieNode()  # 根节点初始化

    # 向 Trie 树中添加一个单词
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    # 检查 Trie 树中是否存在一个完整的单词
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    # 检查 Trie 树中是否有以指定前缀开始的单词
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# 使用词典构建 Trie 树
def build_trie(dictionary):
    trie = Trie()
    for word in dictionary:
        trie.insert(word)
    return trie

# 使用 Trie 树来全切分一个句子
def all_cut(sentence, trie):
    # 使用深度优先搜索来查找所有可能的词语切分方式
    def dfs(start, path, result):
        # 如果达到句子末尾，则记录当前切分路径
        if start == len(sentence):
            result.append(path[:])
            return
        # 从当前位置开始，尝试每一种切分方式
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            if trie.search(word):
                path.append(word)
                dfs(end, path, result)
                path.pop()

    result = []
    dfs(0, [], result)
    return result

# 构建 Trie
trie = build_trie(Dict.keys())
sentence = "经常有意见分歧"
# 生成所有可能的切分方式
result = all_cut(sentence, trie)
print(result)
