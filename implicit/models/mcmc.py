import torch
from functorch import jacrev, jvp, vjp, vmap
from gpytorch.utils import linear_cg


class LangevinMC:
    """Class for running Langevin MC.

    Primary method is `sample_chain`, which runs MCMC on a batch of initial states.

    Args:
        energy: the energy function with which to sample
    """

    def __init__(self, energy):
        self.energy = energy

    def sample_momentum(self, q):
        """Samples a momentum vector for a step of CHMC.

        Args:
            q: the current position (minibatch, *)

        Returns:
            p: the momentum
        """
        return torch.randn(*q.shape).to(q.device) # Sample initial momentum from gaussian


    def simulate_langevin_step(self, q, p, eps=0.005, grad_clamp=0.03, alpha=None):
        """Simulates a step of Langevin MC.

        Args:
            q: the current position (minibatch, *)
            p: the current momentum (minibatch, *)
            eps: the scaling factor for the momentum
            alpha: the step size; defaults to eps**2 in the case of "proper" Langevin diffusion,
                but is typically tuned separately for high-dimensional datasets

        Returns:
            qs: the new position
        """
        if alpha is None:
            alpha = eps ** 2

        with torch.no_grad():
            # Need function for grad to take in single image and yield scalar
            single_im_energy = lambda x: self.energy(x[None,...]).squeeze()
            energy_grad = vmap(jacrev(single_im_energy))(q)
            if grad_clamp is not None:
                energy_grad = energy_grad.clamp(-grad_clamp, grad_clamp)

            # Take Langevin dynamics step
            qs = q + eps*p - (alpha/2)*energy_grad

        return qs.detach()

    def sample_chain(self, q0, n_steps=20, **step_kwargs):
        """Samples Markov chain from an initial state.

        Todo:
            Compress kwargs into dict

        Args:
            q0: the initial state (minibatch, *)
            n_steps: the number of Langevin dynamics steps
            step_kwargs: keyword arguments to be passed to self.simulate_langevin_step; eg.
                eps: the scaling factor for the momentum
                alpha: the step size; defaults to eps**2 in the case of "proper" Langevin diffusion,
                    but is typically tuned separately for high-dimensional datasets
                opt_cls: the optimizer class in torch.optim (if performing constrained MCMC)
                opt_steps: the maximum number of steps for the simulator's optimizer (if
                    performing constrained MCMC)

        Returns:
            The end state of the chain.
        """
        q = q0

        for _ in range(n_steps):
            p = self.sample_momentum(q)
            qs = self.simulate_langevin_step(q, p, **step_kwargs)
            q = qs.detach()

        return q.detach()


class ConstrainedLangevinMC(LangevinMC):
    """Class for running constrained Langevin MC.

    Primary method is `sample_chain`, which runs MCMC on a batch of initial states.

    Args:
        mdf: a manifold-defining function (the manifold is given by M = {x: mdf(x) = 0})
        energy: the energy function with which to sample
    """

    def __init__(self, mdf, energy):
        self.mdf = mdf
        self.energy = energy

    def sample_momentum(self, q):
        """Samples a momentum vector for a step of CHMC.

        Args:
            q: the current position (minibatch, *)

        Returns:
            p: the momentum
        """
        p0 = torch.randn(*q.shape).to(q.device) # Sample initial momentum from gaussian

        # Project p0 onto tangent space
        with torch.no_grad():
            _, Jp0 = jvp(self.mdf, (q,), (p0,))
            Jp0 = Jp0[...,None] # Add dimension for linear_cg func

            def jtj_closure(vec):
                vec = vec[:,:,0] # Squeeze b x (n-m) x 1 input

                _, vjp_fn = vjp(self.mdf, q)
                JTv, = vjp_fn(vec)
                _, JJTv = jvp(self.mdf, (q,), (JTv,))
                return JJTv[...,None] # Add back dimension

            JJT_inv_Jp0 = linear_cg(jtj_closure, Jp0)[:,:,0]

            _, vjp_fn = vjp(self.mdf, q)
            JdagJp0, = vjp_fn(JJT_inv_Jp0)

            p = p0 - JdagJp0

        return p

    def simulate_langevin_step(self, q, p, eps=0.005, alpha=None, grad_clamp=0.03,
                               opt_cls="LBFGS", opt_steps=100, **opt_kwargs):
        """Simulates a step of Constrained Langevin MC.

        Args:
            q: the current position (minibatch, *)
            p: the current momentum (minibatch, *)
            eps: the scaling factor for the momentum
            alpha: the step size; defaults to eps**2 in the case of "proper" Langevin diffusion,
                but is typically tuned separately for high-dimensional datasets
            opt_cls: the optimizer class in torch.optim to use to solve the step
            opt_steps: the maximum number of steps for the optimizer
            opt_kwargs: keyword arguments passed to optimizer upon instantiation

        Returns:
            qs: the new position
        """
        if alpha is None:
            alpha = eps ** 2

        with torch.no_grad():
            # Need function for grad to take in single image and yield scalar
            single_im_energy = lambda x: self.energy(x[None,...]).squeeze()
            energy_grad = vmap(jacrev(single_im_energy))(q).clamp(-grad_clamp, grad_clamp)
            out, vjp_fn = vjp(self.mdf, q)

        def langevin_step(lamb):
            JTlambda = vjp_fn(lamb)[0]
            return q + eps*p - (alpha/2)*energy_grad - (alpha/2)*JTlambda

        # Solve for langevin step that remains on manifold
        lamb = torch.zeros(out.shape).to(q.device)
        lamb.requires_grad = True

        opt_cls = getattr(torch.optim, opt_cls)
        lamb_opt = opt_cls([lamb], **opt_kwargs)

        for i in range(opt_steps):
            def closure():
                lamb_opt.zero_grad()
                loss = self.mdf(langevin_step(lamb)).square().sum()
                loss.backward()
                return loss

            loss = lamb_opt.step(closure)

        return langevin_step(lamb).detach()
