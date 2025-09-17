# 定义几个简单的函数
def greet():
    print("Hello!")

def farewell():
    print("Goodbye!")

def ask():
    print("How are you?")

# 将函数名（引用）存储在列表中
functions = [greet, farewell, ask]

# 遍历列表并调用函数
for func in functions:
    func()  # 调用函数


# 定义带参数的函数
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

# 将函数名存储在字典中
operations = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply
}

# 动态调用函数
operation_name = "add"  # 假设用户选择加法
if operation_name in operations:
    result = operations[operation_name](10, 5)  # 调用add(10, 5)
    print("Result:", result)
else:
    print("Invalid operation!")