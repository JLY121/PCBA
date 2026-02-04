import sys
sys.path.append("../")

# 导入所有攻击方法
from fl_utils.attacker import Our_Attacker


class AttackerDispatcher:
    def __init__(self, helper):
        self.helper = helper
        self.attacker = None
        self.setup_attacker()

    def setup_attacker(self):

        if  self.helper.config.attack_type == 'Our':
            self.attacker = Our_Attacker(self.helper)
        elif self.helper.config.attack_type == 'None':
            self.attacker = None
        else:
            raise NotImplementedError(f"Attack method '{self.helper.config.attack_type}' not implemented")

    def get_attacker(self):
        """
        返回初始化好的攻击者对象
        """
        print(f"初始化攻击者: {self.helper.config.attack_type}")
        return self.attacker
