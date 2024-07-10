# week3作业

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常": 0.1,
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
		"分": 0.1}

# 待切分文本
sentence = "经常有意见分歧"
sl = list(sentence)
# state_dict = {}
# for word in Dict:
# 	for end in range(1, 4):
# 		if word[0:end] in Dict:
# 			state_dict[word[0:end]] = 1
# 		else:
# 			state_dict[word[0:end]] = 0
#
# print(state_dict)

state_dict = {}
for word in Dict:
	state_dict[word] = 1
print(state_dict)


length = len(sentence)
idx = 0
sen_list = []
for _ in range(length):
	sen_list.append(_)

str = sentence
def cut_mod(str,state_dict,Max_len = 3):
	log_all = []
	for i in range(1,Max_len+1):
		m_len = i
		log= []
		str = sentence
		while str != '': #注意 这里表示不是空字符串时，中间不写任何东西 没有空格。is not None 是表示不是空值，不止针对字符串，!= 0 也是非空值
			lens = min(m_len,len(str))
			word = str[:lens]
			while word not in state_dict:
				if len(word) == 1:
					break
				word = word[:len(word) - 1]
			log.append(word)
			str = str[len(word):]
		log_all.append(log)
	return log_all


print(cut_mod(str,state_dict))



