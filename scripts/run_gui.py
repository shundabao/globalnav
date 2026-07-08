#!/usr/bin/env python3
"""Launch GLOBALNAV interactive GUI."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from globe_nav.gui.app import main

if __name__ == '__main__':
    main()
