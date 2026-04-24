import argparse
from pathlib import Path
from typing import TypeVar, Type, List, cast

from omegaconf import OmegaConf

T = TypeVar("T")


def parse_args(ctype: Type[T], args=None) -> T:
    """
    CLI interface function for laod_config
    sample: python test.py --config conf.yaml test.a=1 test.b=false
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None, help="yaml config file")
    parser.add_argument("overrides", nargs="*", default=[], help="config file overrides")
    args = parser.parse_args(args)
    return load_config(ctype, args.config, args.overrides)


def load_config(ctype: Type[T], config: Path, overrides: List[str]) -> T:
    """
    load configuration from yaml config file & CLI overrides, based on CLI input
    precedence: CLI override > yaml config > dataclass default
    """
    confs = []
    confs.append(OmegaConf.structured(ctype))
    if config:
        confs.append(OmegaConf.load(config))
    if overrides:
        confs.append(OmegaConf.from_dotlist(overrides))
    conf = OmegaConf.merge(*confs)
    return cast(T, conf)
