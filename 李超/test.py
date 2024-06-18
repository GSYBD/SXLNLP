#使用while循环打印从1到5的数字
# i = 1
# while i <= 5:
#     print(i)
#     i += 1
#使用while训练计算1-100的偶数和
# num = 2
# save = 0
# while save <= 100:
#     save += num
#     num += 2
# print('计算1-100的偶数和:' , +save)
# num = 1  # 初始化计数器变量num
# sum_even = 0  # 初始化偶数和的变量sum_even
#
# while num <= 100:
#     if num % 2 == 0:  # 判断num是否为偶数
#         sum_even += num  # 如果是偶数，将其累加到sum_even中
#     num += 1  # 将计数器num加1
#
# print("1到100之间的偶数和为:", sum_even)
#3. 模拟用户登录给三次机会
# i = 0
# while i < 3:
#     username =input('请输入用户名:')
#     password = input('请输入密码:')
#     进行用户认证的逻辑判断，
#     if username=='user' and password=='password':
#         print('登录成功')
#         break;  # 登录成功，跳出循环
#     else:
#         print('登录失败！')
#         i += 1 # 计数器加1
# if i == 3:
#     print('登录失败次数过多，账号已锁定')
#猜数字游戏
import random
#是否继续玩的标志
play_again = True
while play_again:
    # 生成1到100之间的随机整数
    secret_number = random.randint(1,1000)
    # 猜测次数计数器
    attempts = 0;
    while attempts < 10:
        guess = int(input("请猜一个1到1000之间的数字: "))
        if guess == secret_number:
            print("恭喜你，猜对了！")
            play_again = False  # 猜对了，不再询问是否继续玩
            break
        elif guess < secret_number:
            print("猜的数字太低了！")
        else:
            print("猜的数字太高了！")

        attempts += 1  # 猜测次数加1

    if attempts == 10:
        choice = input("猜测次数已达上限，是否还想继续玩？(Y/N): ")
        if choice.lower() == "n":
            play_again = False  # 不再继续玩