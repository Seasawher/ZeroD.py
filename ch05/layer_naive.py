class MulLayer:
    """乗算レイヤー"""
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        """順伝播"""
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        """逆伝播"""
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy