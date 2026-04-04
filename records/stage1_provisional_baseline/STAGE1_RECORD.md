# Stage-1 Provisional Baseline 复现记录

## 1. 本轮目标
完成远程 Linux 上 Stage-1 baseline 的训练、测试与可视化闭环。

## 2. 本轮关键改动
- val/test 改为确定性采样，降低评估随机性；
- 保持当前训练/测试/可视化预处理一致；
- 增加基于 topk_tiles.csv 生成热力图/热点图的补充脚本（如有）。

## 3. 本轮数据状态
- patient 去重后构建约 100 张 slide 的样本池；
- 使用 patient-level split；
- 当前 split 的 train/val/test 均包含双类。

## 4. 本轮结果
### Validation
- AUC = 0.8571
- F1 = 0.6667
- Sensitivity = 1.0000
- Specificity = 0.0000

### Test
- AUC = 0.5714
- F1 = 0.6364
- Sensitivity = 0.7778
- Specificity = 0.1429

## 5. 结果解释
- 工程流程已跑通；
- baseline 已具备一定排序能力；
- 但 test 泛化不稳，specificity 偏低，说明当前版本仍属于 provisional baseline。

## 6. 代表性可视化结论
- Top-K tiles 显示模型更偏向关注间质/纤维样区域；
- FN 中 attention 分布较平，说明模型未能在部分阳性 slide 上找到强阳性证据。

## 7. 当前结论
- 不建议立即进入第二阶段正式主动学习实验；
- 先完成阶段一稳定性确认与代码归档。