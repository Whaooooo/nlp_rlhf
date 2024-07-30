def read_key(service, name="default"):
    with open(f"/home/zzo/.cache/data/marl/keys/{service}/{name}", "r") as f:
        return f.read().strip()
