import numpy as np
from scipy.stats import pearsonr
import sys
import logging

LOG_LEVEL = logging.DEBUG
#LOG_LEVEL = logging.INFO
# create logger
logger = logging.getLogger('hackerrank')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(LOG_LEVEL)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

#  ----------------- Main --------------------
state = 0
logger.debug("Correlation and Regression Lines - A Quick Recap #1")
for line in sys.stdin:
    if state == 0:
        # Read physical score
        phy_score = line.split('  ')
        phy_arr = np.asarray(phy_score[1:], dtype=float)
        logger.debug(str(phy_score[1:]))
        logger.debug(str(phy_arr))
        state = 1
    elif state == 1:
        his_score = line.split('  ')
        his_arr = np.asarray(his_score[1:], dtype=float)
        logger.debug(str(his_score[1:]))
        logger.debug(str(his_arr))
        break
correlation = pearsonr(phy_arr, his_arr)
logger.debug(str(correlation))
print(str(correlation))