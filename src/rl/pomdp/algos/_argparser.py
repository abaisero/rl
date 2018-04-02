from .gpomdp import GPOMDP
from .isgpomdp import IsGPOMDP
from .expgpomdp import ExpGPOMDP

from .cfgpomdp import CFGPOMDP
import argparse


# TODO implement using some type of decorator directly!?

parser = argparse.ArgumentParser(description='Algo')
_subparsers = parser.add_subparsers(title='Algo', dest='algo')
_subparsers.required = True

_parser = GPOMDP.parser
_subparsers.add_parser('gpomdp', parents=[_parser], help='GPOMDP')

_parser = IsGPOMDP.parser
_subparsers.add_parser('isgpomdp', parents=[_parser], help='IsGPOMDP')

_parser = ExpGPOMDP.parser
_subparsers.add_parser('expgpomdp', parents=[_parser], help='ExpGPOMDP')

_parser = CFGPOMDP.parser
_subparsers.add_parser('cfgpomdp', parents=[_parser], help='CFGPOMDP')
