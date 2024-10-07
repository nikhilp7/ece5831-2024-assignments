from logic_gate import LogicGate

gate = LogicGate()

xs = [(0, 0), (0, 1), (1, 0), (1, 1)]

for x in xs:
    y = gate.andd(x[0], x[1])
    print('{} AND {} = {}'.format(x[0], x[1], y))
    
for x in xs:
    y = gate.nand(x[0], x[1])
    print('{} NAND {} = {}'.format(x[0], x[1], y))
    
for x in xs:
    y = gate.orr(x[0], x[1])
    print('{} OR {} = {}'.format(x[0], x[1], y))
    
    
for x in xs:
    y = gate.nor(x[0], x[1])
    print('{} NOR {} = {}'.format(x[0], x[1], y))
    
for x in xs:
    y = gate.xor(x[0], x[1])
    print('{} XOR {} = {}'.format(x[0], x[1], y))