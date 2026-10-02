# 参与贡献

[English](CONTRIBUTING.md)

小而完整的改动最容易评审。可直接提 PR 修文档、补离线示例或供应商适配；
大功能先开 issue 描述使用者、输入、预期输出与验证方法。

## 本地检查

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m judgekit demo
python -m judgekit demo --lang zh
python benchmarks/run_bench.py --models rules --limit 4
```

默认测试不需要 key，也不应访问远端 API。依赖本机研究产物的检查会显式跳过；
CI 使用 Ubuntu / Windows 与 Python 3.10 / 3.12。
行为改动应添加能捕获原问题的离线回归检查；只改文字时核对命令、链接和两种语言的一致性。

改动 CLI 或打包时，再构建并在源码目录外检查安装后的 wheel：

```bash
python -m pip wheel --no-deps --wheel-dir dist .
python tools/check_wheel.py
```

检查器创建临时虚拟环境，复用前面安装的 PyYAML，并用 `--no-index --no-deps` 安装 wheel。
通过命令行入口与 `python -m judgekit` 检查两种语言，演示期间阻止网络连接与子进程。
CI 对每个支持的 OS/Python 组合运行此检查。安装包可能下载构建依赖；演示和默认测试不调用付费 API。

## 贡献类型

- **文档与翻译**：保留命令、路径、数字、数据集版本和局限；更新中英两版，不能把历史结果写成当前测量。
- **示例与 recipe**：提供可运行的本地零成本路径。涉及人的建议由人作决定，求职工具保留意愿问卷，避免企业自动筛除。
- **供应商**：实现 `decide(task, x) -> Decision` 并在注册表接入；提交 mock HTTP 测试，包含失败行为、实际 provider 与费用计算。
- **评测数据**：注明来源、许可、采样方法、标签定义、数据版本与数量；将训练/阈值选择集和测试集分开。没有发布许可或含个人信息的原文不要加入公共仓库。

## 结果与费用

报告应注明模型 ID、日期、数据修订、提示词、分母、失败数和是否发生规则兜底。
小样本准确率附置信区间；同一数据集上选的阈值必须标注 in-sample。
订阅边际费用、API 花费、硬件与电力不是同一个指标。
当前与历史口径见 [评测证据](docs/evaluation.md)。

## 提交边界

不要提交 `.env`、key、私人简历、个人数据、原始模型响应缓存、内部状态或整份运行日志。
只暂存明确检查过的文件；共享工作目录可能含他人的未提交研究成果。
PR 描述说明问题、改后行为、实际检查与未验证范围即可。
