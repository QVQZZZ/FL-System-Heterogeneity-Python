class Model:
    """
    Model类
    成员变量:
        width_factor:模型的尺寸
        parameters:真正的模型,是一个nn.Module实现类的实例
    """

    def __init__(self, width_factor, parameters):
        self.width_factor = width_factor
        self.parameters = parameters

    def __str__(self):
        return f"Model(width_factor={self.width_factor}, parameters={self.parameters})"
