import numpy as np

class LogicGate:
    def __init__(self):
        pass
    
    def andd(self, x1, x2):
        x = np.array([x1, x2])
        w = np.array([0.8, 0.8])
        b = -0.9
        
        y = np.sum(x*w) + b
        
        return 1 if y > 0 else 0
    
    def nand(self, x1, x2):
        return 1 if self.andd(x1, x2) == 0 else 0
    
    def orr(self, x1, x2):
        x = np.array([x1, x2])
        
        w = np.array([0.8, 0.8])
        b = -0.4
        
        y = np.sum(x*w) + b
        
        return 1 if y > 0 else 0
    
    def nor(self, x1, x2):
        return 1 if self.orr(x1, x2) == 0 else 0
    
    def xor(self, x1, x2):
        y1 = self.orr(x1, x2)
        y2 = self.nand(x1, x2)
        y = self.andd(y1, y2)
        
        return y
        


if __name__ == "__main__":
    
    print('Testing Logic Gates:')
    print('AND Gate output for inputs (1, 0):')
    gate = LogicGate()
    y = gate.andd(1, 0)
    print(y)
    
    print('OR Gate output for inputs (1, 0):')
    y = gate.orr(1, 0)
    print(y)
    
    print('NOR Gate output for inputs (1, 0):')
    y = gate.nor(1, 0)
    print(y)
    
    print('NAND Gate output for inputs (1, 0):')
    y = gate.nand(1, 0)
    print(y)
    
    print('XOR Gate output for inputs (1, 0):')
    y = gate.xor(1, 0)
    print(y)
