# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.


import csv
from pprint import pformat
import logging

logger = logging.getLogger(__name__)


class CSVLogger:
    def __init__(self, fnm, col_names):
        logger.info('Creating data logger at {}'.format(fnm))
        self.fnm = fnm
        self.col_names = col_names
        with open(fnm, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(col_names)
        # hold over previous values if empty
        self.vals = {name: None for name in col_names}

    def log(self, **cols):
        self.vals.update(cols)
        logger.info(pformat(self.vals))
        if any(key not in self.col_names for key in self.vals):
            raise Exception('CSVLogger given invalid key')
        with open(self.fnm, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([self.vals[name] for name in self.col_names])
