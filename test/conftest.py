import sys
import os
import pytest

# Add the parent directory to the path so pytest can find our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 