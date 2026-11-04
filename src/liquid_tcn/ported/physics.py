"""Torch reductions with world inertial velocities and base-origin wrenches."""
import torch


def rotate(q, v, inverse=False):
    """wxyz quaternion; broadcasting over particles is explicit at the caller."""
    xyz = -q[..., 1:] if inverse else q[..., 1:]
    xyz, v = torch.broadcast_tensors(xyz, v)
    t = 2*torch.linalg.cross(xyz, v)
    return v + q[..., :1]*t + torch.linalg.cross(xyz, t)


def shift_moment(force, moment_A, x_A, x_O):
    # Moment transport: tau_O = tau_A + (x_A-x_O) x F. The same identity
    # applies to angular impulses when F denotes a linear impulse.
    return moment_A + torch.linalg.cross(x_A-x_O, force)


def aggregates(x, u, mass, xO, q, uO, omega):
    """x/u [B,P,3], mass [B,P]; use actual solver masses, no spacing estimate."""
    m = mass.sum(-1, keepdim=True)
    xc = (mass[..., None]*x).sum(-2)/m
    uc = (mass[..., None]*u).sum(-2)/m
    # Internal angular momentum about the mass-weighted COM:
    # S_W = sum_i (x_i-x_COM) x m_i (u_i-u_COM). This excludes orbital
    # angular momentum, which is added separately for fixed-reference losses.
    S = torch.linalg.cross(x-xc[:, None], mass[..., None]*(u-uc[:, None])).sum(-2)
    r = rotate(q, xc-xO, True)
    # Differentiate r_B = R^T(x_COM-x_O): subtract omega_B x r_B to
    # account for frame rotation. omega passed to this helper is in WORLD.
    v = rotate(q, uc-uO, True)-torch.linalg.cross(rotate(q, omega, True), r)
    return dict(mass=m[:, 0], com_W=xc, com_vel_W=uc, momentum_W=m*uc,
                angular_momentum_world_origin_W=torch.linalg.cross(x, mass[..., None]*u).sum(-2),
                r_B=r, v_B=v, S_B=rotate(q, S, True))


def coupling_impulse(x, before, after, mass, origin):
    # The fluid receives m*(after-before); Newton's third law gives the
    # opposite liquid-on-robot impulse. Despite the local name force, this
    # function returns impulses, with no division by a timestep.
    reaction = mass[..., None]*(before-after)
    force = reaction.sum(-2)
    moment_world = torch.linalg.cross(x, reaction).sum(-2)
    return force, shift_moment(force, moment_world, torch.zeros_like(origin), origin), moment_world


class CausalClock:
    """Right-continuous completed-interval estimates; reset independently per env."""
    def __init__(self, velocities, feature_ticks):
        if feature_ticks < 1:
            raise ValueError('feature_ticks must be positive')
        self.feature_ticks = feature_ticks
        self.anchor = velocities.clone()
        self.held = torch.zeros_like(velocities)
        self.age = torch.zeros(len(velocities), dtype=torch.long, device=velocities.device)
        self.valid = torch.zeros_like(self.age, dtype=torch.bool)
        self.episode = torch.zeros_like(self.age)

    def reset(self, ids, velocities):
        self.anchor[ids] = velocities[ids]
        self.held[ids] = 0
        self.age[ids] = 0
        self.valid[ids] = False
        self.episode[ids] += 1

    def advance(self, velocities, dt):
        self.age += 1
        event = self.age.remainder(self.feature_ticks) == 0
        # Update only after a complete feature interval. Between events the
        # last completed average stays held; reset entries remain invalid until
        # their own first interval completes, independently of other environments.
        self.held[event] = (velocities[event]-self.anchor[event])/(dt*self.feature_ticks)
        self.anchor[event] = velocities[event]
        self.valid[event] = True
        return event


def isolated_window(episode_ids, valid):
    """Mask [T,B] windows crossing any reset or invalid state."""
    return valid.all(0) & (episode_ids == episode_ids[:1]).all(0)
