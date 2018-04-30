from .reinforce import REINFORCE
import argparse


# TODO implement using some type of decorator directly!?

parser = argparse.ArgumentParser(description='Algo')
_subparsers = parser.add_subparsers(title='Algo', dest='algo')
_subparsers.required = True

_parser = REINFORCE.parser
_subparsers.add_parser('reinforce', parents=[_parser], help='REINFORCE')
