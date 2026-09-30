# 官方实现的参考副本（仅供等价性测试）

基线的关键组件逐一与这些**未经修改**的官方代码做数值比对，
而不是只看"能跑通"。唯一的改动是把 USB 对其内部基类的 import
替换为一个空替身，因为该基类只提供 hook 注册机制，不参与计算。

| 文件 | 来源 | 许可 |
|---|---|---|
| `wb_class_balanced_loss.py` | ShadeAlsha/LTR-weight-balancing `utils/class_balanced_loss.py` | MIT |
| `wb_regularizers.py` | ShadeAlsha/LTR-weight-balancing `utils/regularizers.py` | MIT |
| `usb_flexmatch_utils.py` | microsoft/Semi-supervised-learning `semilearn/algorithms/flexmatch/utils.py` | MIT |
