from datetime import datetime
from datetime import timedelta

import numpy as np

# Stone Soup imports:
from stonesoup.types.state import State, GaussianState
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, ConstantVelocity)
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.updater.kalman import UnscentedKalmanUpdater
from stonesoup.predictor.kalman import UnscentedKalmanPredictor
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.tracker.simple import MultiTargetTracker
from matplotlib import pyplot as plt

# Define the simulation start time

from stonesoup.platform.base import FixedPlatform

# Define the initial platform position, in this case the origin
def run(gen):
    platform.add_sensor(gen)
    platform_state_vector = StateVector([[0], [0]])
    position_mapping = (0, 1)

    # Create the initial state (position, time), notice it is set to the simulation start time defined earlier
    platform_state = State(platform_state_vector, 0)

    # create our fixed platform
    platform = FixedPlatform(states=platform_state,
                            position_mapping=position_mapping)


