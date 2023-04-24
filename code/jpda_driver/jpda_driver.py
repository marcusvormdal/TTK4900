import numpy as np
from datetime import timedelta, datetime

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import JPDA
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.types.state import GaussianState
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.tracker.simple import MultiTargetMixtureTracker
from stonesoup.types.state import GaussianState
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.deleter.time import UpdateTimeStepsDeleter

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005)])
measurement_model = LinearGaussian(
    ndim_state=4, mapping=[0,1,2,3], noise_covar=np.diag([1**2, 1**2, 1**2, 1**2]))

predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(measurement_model)

hypothesiser = PDAHypothesiser(predictor=predictor,
                               updater=updater,
                               clutter_spatial_density=0.125,
                               prob_detect=0.9)

data_associator = JPDA(hypothesiser=hypothesiser)

deleter_init = UpdateTimeStepsDeleter(time_steps_since_update=3)
deleter = UpdateTimeStepsDeleter(time_steps_since_update=15)

prior_state=GaussianState(StateVector(np.zeros((4,1))),
                            CovarianceMatrix(np.diag([100**2, 30**2, 100**2, 30**2])))

initiator = MultiMeasurementInitiator(
    prior_state,
    measurement_model=measurement_model,
    deleter=deleter_init,
    data_associator=data_associator,
    updater=updater,
    min_points=2,
    )


tracks, all_tracks = set(), set()

def track(detector_generator):
    tracker = MultiTargetMixtureTracker(
        initiator=initiator,
        deleter=deleter,
        detector=detector_generator,
        data_associator=data_associator,
        updater=updater,
    )
    
    return tracker