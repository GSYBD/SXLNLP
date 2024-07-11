import json
import time

# vacob ={
#     '我':1 ,
#     '经常':1,
#     '有':1,
#     '目的':1,
#     '有目的':1,
#     '的':1,
#     '去':1,
#     '看':1,
#     '看电影':1,
#     '电影'   :1
# }
vacob ={
"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1
}

str ='经常有意见分歧'
# 分词表json化，方便读取
vacob_json = json.dumps(vacob)

# 获取分词窗口的最大宽度
keys_len = [len(key) for key in vacob.keys()]
max_key_len = max(keys_len)  

# 主分词数组(构建的目标切词数组)，记录结果过程
# 与最后输出分词拆分结果结构一致，是2维数组
main_vacob = [[]]

# 字符串全拆分的可能情况数，初始为0
i = 0

# 对main_vacob的迭代次数
j = 0

# 第二层：对分词窗口进行遍历
# newRow：用于控制切词数组是否需要新加行，false：新加行，true：不加
# sourceStr：原始的待切分字符串
# start_ind,end_ind：分词窗口索引
def constructMainVacob(main_vacob_l, sourceStr, start_ind,end_ind, newRow = False):
    global i , main_vacob
    if sum([len(_) for _ in main_vacob[i]]) == len(sourceStr):
        i += 1
    
    # 遍历到最后原字符串最后一个词时，输出遍历结果，结束遍历
    if end_ind > len(sourceStr):
        return

    # 截取目标词
    dest_vac = sourceStr[start_ind:end_ind]

    # 比较截取分词长度和分词窗口长度
    if len(dest_vac) <= max_key_len: 
        if(vacob.get(dest_vac, 0)):            
            # 如果分词窗口截取的词在词表中，则构建主分词数组
            # print("second ite ,dest_vac: \033[1;31m %s \033[0m ,start_ind:%d , end_ind:%d " %(dest_vac, start_ind,end_ind))

            if not newRow:
                # 如果是当前分词窗口第一次取到有效的分词
                main_vacob_l.append(dest_vac)
                newRow = True

            else:
                # 只有构建了新的main_vacob行以后，才需要累加i，以控制对main_vacob数组的遍历
                # 如果是当前分词窗口的后续迭代中找到了有效的分词，则主分词数组中新建一行用来存储
                main_vacob.append(main_vacob_l[:])
                # print("second ite ,i:%d ,main_vacob : %s"%(i,main_vacob))
                # 注意：这里必须截断当前分词窗口截取到的有效词
                # 同时，将找到的有效分词追加到新建行中
                # print("second ite ,main_vacob[len(main_vacob)-1][len(main_vacob_l)-1]: " ,main_vacob[len(main_vacob)-1][len(main_vacob_l)-1])
                main_vacob[len(main_vacob)-1][len(main_vacob_l)-1] = dest_vac

                # print("second ite ,i:%d ,main_vacob : %s"%(i,main_vacob))
                # print("second ite ,main_vacob[len(main_vacob)-1]: " ,main_vacob[len(main_vacob)-1])
            
        # end if(vacob.get(dest_vac, 0)): 

        # 仅仅是移动分词窗口右边界，保持左边界不动，进行下一次迭代，寻找更长词的可能性
        end_ind += 1
        # print("newRow = False:", newRow)
        constructMainVacob(main_vacob_l, sourceStr,start_ind,end_ind,newRow)       

    else :
        # 分词窗口遍历完成
        # 超过分词窗口最大长度，返回上一层开始处理待分词串上剩余的字符串
        pass

# 对主分词串进行遍历
def iterStr(sourceStr,start_ind, end_ind):
    # time.sleep(1)
    global j
    j+=1
    
    # print("%d ==============================len(main_vacob): %d , end_ind:%d" %(i,len(main_vacob), end_ind))
    # print("start_ind = sum([ len(_) for _ in main_vacob[i]]):  " , start_ind)
    end_ind =  start_ind + 1
    # print("main_vacob[i] : %s, %d"%( main_vacob[i],i))
    constructMainVacob(main_vacob[i] , sourceStr, start_ind ,end_ind)

    start_ind = sum([ len(_) for _ in main_vacob[i]])
    # print("next start_ind:", start_ind)
    if (i == len(main_vacob) - 1 and start_ind == len(sourceStr)):
        j+=1
        if(j > 2):
            return
    iterStr(sourceStr,start_ind ,start_ind+ 1)


if __name__ == "__main__":
    iterStr(str, 0,1)
    print(main_vacob)