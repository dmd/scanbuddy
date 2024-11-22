#!/usr/bin/env python3 -u

import sys
import time
import logging
from pubsub import pub
from pathlib import Path
from argparse import ArgumentParser
import multiprocessing as mp
from scanbuddy.watcher.directory import DirectoryWatcher
from scanbuddy.proc import Processor
from scanbuddy.proc.volreg import VolReg
from scanbuddy.proc.params import Params
from scanbuddy.proc.snr import SNR
from scanbuddy.view.dash import View
from scanbuddy.broker.redis import MessageBroker
from scanbuddy.config import Config

logger = logging.getLogger('main')
FORMAT = '%(process)d %(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
mp.set_start_method('fork')

def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--mock', action='store_true')
    parser.add_argument('-c', '--config', required=True, type=Path)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--snr-interval', default=10, 
        help='Every N volumes snr should be calculated')
    args = parser.parse_args()

    config = Config(args.config)

    cpus = 1
    pool = mp.Pool(cpus)

    broker = MessageBroker()
    watcher = DirectoryWatcher(args.folder, pool)
    processor = Processor()
    params = Params(
        broker=broker,
        config=config
    )
    volreg = VolReg(mock=args.mock)
    snr = SNR()
    view = View(
        host=args.host,
        port=args.port,
        config=config,
        debug=args.verbose
    )

    if args.verbose:
        logging.getLogger('scanbuddy.proc').setLevel(logging.DEBUG)
        logging.getLogger('scanbuddy.proc.params').setLevel(logging.DEBUG)
        logging.getLogger('scanbuddy.proc.volreg').setLevel(logging.DEBUG)
        logging.getLogger('scanbuddy.view.dash').setLevel(logging.DEBUG)
   
    # logging from this module is useful, but noisy
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    # start the watcher and view
    watcher.start()
    view.forever()

if __name__ == '__main__':
    main()
