  def __init__(self, model_path):
     # 初始化函数
     self.device_id = 0
     # step1: 初始化
     ret = acl.init()
     # 指定运算的Device
     ret = acl.rt.set_device(self.device_id)
     # step2: 加载模型，本示例为ResNet-50模型
     # 加载离线模型文件，返回标识模型的ID
     self.model_id, ret = acl.mdl.load_from_file(model_path)
     # 创建空白模型描述信息，获取模型描述信息的指针地址
     self.model_desc = acl.mdl.create_desc()
     # 通过模型的ID，将模型的描述信息填充到model_desc
     ret = acl.mdl.get_desc(self.model_desc, self.model_id)

     # step3：创建输入输出数据集
     # 创建输入数据集
     self.input_dataset, self.input_data = self.prepare_dataset('input')
     # 创建输出数据集
     self.output_dataset, self.output_data = self.prepare_dataset('output')
     
