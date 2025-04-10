from dotenv import load_dotenv, find_dotenv
import os

# 查找 .env 文件
dotenv_path = find_dotenv()
if dotenv_path:
    print(f"Found .env file at: {dotenv_path}")
    # 加载并打印，设置 override=True 可以看到加载后的值（如果它覆盖了系统变量）
    # verbose=True 会打印出它设置或跳过的变量
    load_dotenv(dotenv_path=dotenv_path, override=True, verbose=True)
else:
    print("No .env file found.")

import pprint # 用于更美观地打印字典

print("--- All Environment Variables ---")
pprint.pprint(dict(os.environ))