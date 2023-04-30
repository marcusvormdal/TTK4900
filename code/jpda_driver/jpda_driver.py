import numpy as np
from datetime import timedelta, datetime

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.dataassociator.probability import JPDA, PDA
from stonesoup.types.state import GaussianState
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.tracker.simple import MultiTargetMixtureTracker
from stonesoup.types.state import GaussianState
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.deleter.error import CovarianceBasedDeleter

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.001),
                                                          ConstantVelocity(0.001)])
print(transition_model)

measurement_model = LinearGaussian(
    ndim_state=4, mapping=[0,2], noise_covar=np.diag([0.5**2, 0.5**2]))

predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(measurement_model)


hypothesiser = PDAHypothesiser(predictor=predictor,
                               updater=updater,
                               clutter_spatial_density=0.125)

data_associator = JPDA(hypothesiser=hypothesiser)

deleter_init = UpdateTimeStepsDeleter(time_steps_since_update=3)
deleter = UpdateTimeStepsDeleter(time_steps_since_update=5)

prior_state=GaussianState(StateVector(np.zeros((4,1))),
                            CovarianceMatrix(np.diag([1**2, 1**2, 1**2, 1**2])))


#deleter_new = CovarianceBasedDeleter(covar_trace_thresh=1)

init_hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=0.3)
init_data_associator = GNNWith2DAssignment(init_hypothesiser)

initiator = MultiMeasurementInitiator(
    prior_state=GaussianState([[0], [0], [0], [0]], np.diag([0, 1, 0, 1])),
    measurement_model=measurement_model,
    deleter=deleter_init,
    data_associator=init_data_associator,
    updater=updater,
    min_points=2,
    )


def track(detector_generator):
    
    tracker = MultiTargetMixtureTracker(
        initiator=initiator,
        deleter=deleter,
        detector=detector_generator,
        data_associator=data_associator,
        updater=updater,
    )
    
    return tracker