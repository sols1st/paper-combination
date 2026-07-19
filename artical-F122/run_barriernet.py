"""
BarrierNet-AEBS Convenience Entry Script

Usage:
    python run_barriernet.py
    python run_barriernet.py --epochs 50 --robust-margin 0.1
    python run_barriernet.py --load-path ./Aebs/BarrierNet/results/barrier_net_aebs.pth
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Aebs.BarrierNet.main import main

if __name__ == '__main__':
    main()
