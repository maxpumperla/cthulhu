#!/usr/bin/env python
# -*- coding: utf-8 -*-
from cthulhu.names import all_names
import os

def replace(folder="./docs"):
    for k, c in all_names.items():
        for dname, dirs, files in os.walk(folder):
            for filepath in files:
                if any(n in filepath for n in ["md", "py", "html"]):
                    filepath = os.path.join(dname, filepath)
                    print(filepath)
                    with open(filepath) as f:
                        s = f.read()
                    s = s.replace(k, c)
                    with open(filepath, "w") as f:
                        f.write(s)

if __name__ == "__main__":
    replace()
