loss = (- losses.rec_ll
        - self._caps_ll_weight * losses.log_prob
        + self._dynamic_l2_weight * losses.dynamic_weights_l2
        + self._primary_caps_sparsity_weight * losses.primary_caps_l1

        # posterior sparsity had no effect in ablation study
        + self._posterior_within_example_sparsity_weight * losses.posterior_within_sparsity_loss
        - self._posterior_between_example_sparsity_weight * losses.posterior_between_sparsity_loss
        + self._prior_within_example_sparsity_weight * losses.prior_within_sparsity_loss
        - self._prior_between_example_sparsity_weight * losses.prior_between_sparsity_loss

        + self._weight_decay * losses.weight_decay_loss
        )