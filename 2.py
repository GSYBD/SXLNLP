# 不定长必选参数
# def sum_num(*args):
#     print(args, type(args))
#     result = 0
#     for value in args:
#         result = result + value
#     return result
def sum_num(**kwargs):
    print(kwargs, type(kwargs))
    result = 0
    for key, value in kwargs.items():
        print(key, value)
        result = result + value
    return result


def sum_num2(*args):
    print(args, type(args))
    result = 0
    for value in args:
        result = result + value


a = {"num1": 1, "num2": 2, "num3": 3}

print(sum_num(a))
