

# 注：配置中，大量字段是早已废弃的历史遗留，这包括：ppm, cond_qk, 两阶段训练




import json
import os
import torch

class ConfigDict(dict):
    def __getattr__(self, name):
        try:
            value = self[name]
            # 如果值是字典，递归地将其也转换为ConfigDict
            if isinstance(value, dict):
                return ConfigDict(value)
            return value
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}', or it was not provided in the config file.")

    def __setattr__(self, name, value):
        self[name] = value

    def save(self, fn):
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        with open(fn, 'w') as f:
            json.dump(self, f, indent=4)

    def load(self, fn):
        if not os.path.exists(fn):
            raise FileNotFoundError(f"Configuration file not found: {fn}")
        with open(fn, 'r') as f:
            self.update(json.load(f))
        # 加载后，确保顶层也是一个ConfigDict实例
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = ConfigDict(v)

    @property
    def device(self):
        try:
            # 确保从 training 或顶层获取设备名称
            dev_name = self.get('training', {}).get('dev_name', 'cuda' if torch.cuda.is_available() else 'cpu')
            return torch.device(dev_name)
        except Exception:
            return torch.device('cpu')

CFG = ConfigDict()