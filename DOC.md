# 环境配置
```
uv venv -p 3.13
uv pip install -e deeppresenter
uv pip install pptagent
```

然后配置 batch_eval_config.yaml 的评分路径，模型和并发，运行 batch_eval.py 。