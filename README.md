# schemion-dev

во всех submodules установлены китайские зеркала для pypi, так как стандартный не работает

```bash
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    --timeout 120 \
    --no-cache-dir -r requirements.txt
```

если надо вернуть то строка будет такой:

```bash
RUN pip install --no-cache-dir -r requirements.txt
```