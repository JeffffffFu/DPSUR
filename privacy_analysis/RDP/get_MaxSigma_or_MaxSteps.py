

from privacy_analysis.RDP.compute_dp_sgd import apply_dp_sgd_analysis


def get_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    steps: int,
    alphas,
    epsilon_tolerance: float = 0.01,
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate
    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        steps: number of steps to run
        epsilon_tolerance: precision for the binary search
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """

    sigma_low, sigma_high = 0, 10

    eps_high, best_alpha = apply_dp_sgd_analysis(sample_rate, sigma_high, steps, alphas, target_delta)

    if eps_high>target_epsilon:
        raise ValueError("The privacy budget is too low. 当前最大的sigma只到10")

    while target_epsilon - eps_high > epsilon_tolerance:
        sigma = (sigma_low + sigma_high) / 2

        eps, best_alpha = apply_dp_sgd_analysis(sample_rate, sigma, steps, alphas,target_delta)

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return round(sigma_high,2)

#
def get_min_sigma(epsilon_budget_for_valid_in_all_updates, epsilon_budget_for_valid_in_one_iter,delta, q, steps,orders):

    min_sigma_for_all_updates=get_noise_multiplier(epsilon_budget_for_valid_in_all_updates,delta,q,steps,orders)
    print("min_sigma_for_all_updates:",min_sigma_for_all_updates)

    min_sigma_for_one_iter=get_noise_multiplier(epsilon_budget_for_valid_in_one_iter, delta, q,1, orders)
    print("min_sigma_for_one_iter:",min_sigma_for_one_iter)

    return max(min_sigma_for_all_updates,min_sigma_for_one_iter)


def get_max_steps(
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    sigma: float,
    alphas,
    epsilon_tolerance: float = 0.01,
) -> int:

    steps_low, steps_high = 0, 100000

    eps_high, best_alpha = apply_dp_sgd_analysis(sample_rate, sigma, steps_high, alphas, target_delta)

    if eps_high < target_epsilon:
        raise ValueError("The privacy budget is too high.")


    while eps_high - target_epsilon > epsilon_tolerance:
        steps = (steps_low + steps_high) / 2
        eps, best_alpha = apply_dp_sgd_analysis(sample_rate, sigma, steps, alphas,target_delta)

        if eps > target_epsilon:
            steps_high = steps
            eps_high = eps
        else:
            steps_low = steps

    return int(steps_high)
