import sqlite3


# 初始化数据库
def init_db():
    conn = sqlite3.connect('user.db')
    cursor = conn.cursor()
    # 创建用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')


init_db()
