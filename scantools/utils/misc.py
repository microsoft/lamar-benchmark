import argparse


def add_bool_arg(parser: argparse.ArgumentParser, name: str,
                 default: bool = False, description: str = ''):
    """ Argparse helper for enable/disable flags.
    Parameters
    ----------
    parser : argparser
        Parser.
    name : str
        Name of argument (will be used as '--name').
    default : bool, optional
        Whether default argument should enabled or disabled, by default False
    description : str, optional
        Description for '--help' option, by default ''
    """

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name,
                       dest=name,
                       action='store_true',
                       help='Enable: ' + description +
                       (' [default]' if default is True else ''))
    group.add_argument('--no-' + name,
                       dest=name,
                       action='store_false',
                       help='Disable: (no ' + name + ')' +
                       (' [default]' if default is False else ''))
    parser.set_defaults(**{name: default})
