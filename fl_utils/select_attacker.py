import sys
sys.path.append("../")

# 导入所有攻击方法
from fl_utils.attacker import A3FL_Attacker, BadNets_Attacker, Chameleon_Attacker, Neurotoxin_Attacker, Our_Attacker, Mirage_Attacker
# from fl_utils.attacker import BC_Layer_Attacker

class AttackerDispatcher:
    def __init__(self, helper):
        self.helper = helper
        self.attacker = None
        self.setup_attacker()

    def setup_attacker(self):
        """
        根据配置选择相应的攻击方法类进行初始化
        """
        if self.helper.config.attack_type == 'A3FL':
            self.attacker = A3FL_Attacker(self.helper)
        elif self.helper.config.attack_type == 'BadNets':
            self.attacker = BadNets_Attacker(self.helper)
        elif self.helper.config.attack_type == 'Chameleon':
            self.attacker = Chameleon_Attacker(self.helper)
        elif self.helper.config.attack_type == 'Neurotoxin':
            self.attacker = Neurotoxin_Attacker(self.helper)
        elif self.helper.config.attack_type == 'Our':
            self.attacker = Our_Attacker(self.helper)
        elif self.helper.config.attack_type == 'Mirage':
            self.attacker = Mirage_Attacker(self.helper)
        # elif self.helper.config.attack_type == 'BC_Layer':
        #     self.attacker = BC_Layer_Attacker(self.helper)
        elif self.helper.config.attack_type == 'None':
            self.attacker = None
        else:
            raise NotImplementedError(f"Attack method '{self.helper.config.attack_type}' not implemented")

    def get_attacker(self):
        """
        返回初始化好的攻击者对象
        """
        print(f"JLY: 初始化攻击者: {self.helper.config.attack_type}")
        return self.attacker
