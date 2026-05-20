import random

def play_once():
    """
    模拟一次游戏：
    正正负 HHT -> A 赢
    正负负 HTT -> B 赢
    """
    seq = ""

    while True:
        coin = random.choice(["H", "T"])
        seq += coin

        # 只需要看最后 3 位
        if seq.endswith("HHT"):
            return "A"
        elif seq.endswith("HTT"):
            return "B"


def simulate(n=100000):
    a_win = 0
    b_win = 0

    for _ in range(n):
        winner = play_once()
        if winner == "A":
            a_win += 1
        else:
            b_win += 1

    print(f"A 赢次数: {a_win}")
    print(f"B 赢次数: {b_win}")
    print(f"A 获胜概率: {a_win / n:.4f}")
    print(f"B 获胜概率: {b_win / n:.4f}")


simulate(100000)