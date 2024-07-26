# dasQt/__init__.py

# __version__ = '0.1.0'

__VERSION__ = '1.0.1'
__AUTHOR__  = 'Zhiyu Zhang'
__INSTRUCTION__ = 'JiLin University'
__DATE__    = '2024-05-05 20h'
__EMAIL__   = 'erbiaoger@gmail.com'
__SITE__    = 'erbiaoger.site'

def about() -> str:
    info = f"""
    version: {__VERSION__}
    author: {__AUTHOR__}
    instruction: {__INSTRUCTION__}
    date: {__DATE__}
    email: {__EMAIL__}
    site: {__SITE__}"""

    return info

print(about())