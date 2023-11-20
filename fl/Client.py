class Client:
    """
    Client类
    成员变量:
        width_factor:客户端的尺寸
        data:该客户端拥有的数据子集
    """

    def __init__(self, width_factor, data):
        self.width_factor = width_factor
        self.data = data

    def __str__(self):
        return f"Client(width_factor={self.width_factor}, data_len={len(self.data)})"
