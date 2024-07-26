from json import load, dump

def loadConfig(pathname: str) -> dict:
    with open(pathname, 'r') as f:
        return load(f)
    
def saveConfig(config: dict, pathname: str) -> None:
    with open(pathname, 'w') as f:
        dump(config, f, indent=4)