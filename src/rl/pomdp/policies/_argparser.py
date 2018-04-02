from .cf import CF
from .fsc import FSC
from .sparse_fsc import SparseFSC
from .structured_fsc import StructuredFSC
import argparse


# TODO implement using some type of decorator directly!?

parser = argparse.ArgumentParser(description='Policy')
_subparsers = parser.add_subparsers(title='Policy', dest='policy')
_subparsers.required = True

# _parser = FSC.parser('fsc')
# _subparsers.add_parser('fsc', parents=[_parser], help='FSC')

# _parser = SparseFSC.parser('fsc_sparse')
# _subparsers.add_parser('fsc_sparse', parents=[_parser], help='Sparse FSC')

# _parser = StructuredFSC.parser('fsc_structured')
# _subparsers.add_parser('fsc_structured', parents=[_parser], help='Structured FSC')

_parser = CF.parser
_subparsers.add_parser('cf', parents=[_parser], help='CF')

_parser = FSC.parser
_subparsers.add_parser('fsc', parents=[_parser], help='FSC')

_parser = SparseFSC.parser
_subparsers.add_parser('fsc_sparse', parents=[_parser], help='Sparse FSC')

_parser = StructuredFSC.parser
_subparsers.add_parser('fsc_structured', parents=[_parser], help='Structured FSC')
