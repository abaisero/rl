from .softmax import Softmax
import argparse

parser = argparse.ArgumentParser(description='Policy')
_subparsers = parser.add_subparsers(title='Policy', dest='policy')
_subparsers.required = True

_parser = Softmax.parser
_subparsers.add_parser('softmax', parents=[_parser], help='Softmax')
