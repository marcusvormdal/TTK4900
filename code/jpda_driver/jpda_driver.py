import numpy as np
from datetime import timedelta

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
                                               ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.probability import PDAHypothesiser
from stonesoup.types.array import StateVectors
from stonesoup.functions import gm_reduce_single
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.dataassociator.probability import JPDA

transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005)])

measurement_model = LinearGaussian(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.array([[0.75, 0],
                          [0, 0.75]])
    )

predictor = KalmanPredictor(transition_model)
updater = KalmanUpdater(measurement_model)

hypothesiser = PDAHypothesiser(predictor=predictor,
                               updater=updater,
                               clutter_spatial_density=0.125,
                               prob_detect=0.9)

data_associator = JPDA(hypothesiser=hypothesiser)


def track(start_time, tracks, measurements):
    for n, measurements in enumerate(measurements):
        hypotheses = data_associator.associate(tracks,
                                            measurements,
                                            start_time + timedelta(seconds=n))

        # Loop through each track, performing the association step with weights adjusted according to
        # JPDA.
        for track in tracks:
            track_hypotheses = hypotheses[track]

            posterior_states = []
            posterior_state_weights = []
            for hypothesis in track_hypotheses:
                if not hypothesis:
                    posterior_states.append(hypothesis.prediction)
                else:
                    posterior_state = updater.update(hypothesis)
                    posterior_states.append(posterior_state)
                posterior_state_weights.append(hypothesis.probability)

            means = StateVectors([state.state_vector for state in posterior_states])
            covars = np.stack([state.covar for state in posterior_states], axis=2)
            weights = np.asarray(posterior_state_weights)

            # Reduce mixture of states to one posterior estimate Gaussian.
            post_mean, post_covar = gm_reduce_single(means, covars, weights)

            # Add a Gaussian state approximation to the track.
            track.append(GaussianStateUpdate(
                post_mean, post_covar,
                track_hypotheses,
                track_hypotheses[0].measurement.timestamp))
            
    return tracks