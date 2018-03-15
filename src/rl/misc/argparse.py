import argparse


class GroupedAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        group,dest = self.dest.split('.',2)
        groupspace = getattr(namespace, group, argparse.Namespace())
        setattr(groupspace, dest, values)
        setattr(namespace, group, groupspace)
