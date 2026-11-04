"""Differentiable Newton–Euler constraints for completed interval-average labels.

All endpoint tensors are [batch, endpoints, ...]; wrench heads and h have one
fewer time entry. R maps B to world. v is dr_B/dt, NOT relative inertial velocity.
F/tau are liquid-on-robot reactions expressed in the ENDPOINT base frame; tau is
about the FIXED endpoint base-origin location. No forecast alignment is implied.

The caller supplies recorded robot states, actual liquid mass, uniform gravity,
nominal interior dimensions, episode IDs and causal history eligibility. The five
predicted heads are r_B, v_B, S_B, F_average_B and tau_average_O_B. All momenta are
computed from those heads. Keep direct supervision of all five alongside losses.
"""
import torch


def quaternion_matrix(q):
    """Normalize stored wxyz quaternions; differentiable, batched B-to-world R."""
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-30)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def mv(R, v):
    return (R @ v.unsqueeze(-1)).squeeze(-1)


def _result(residual, scales, valid):
    normalized = residual / scales.unsqueeze(-1)
    masked = torch.where(valid[..., None, None], normalized, torch.zeros_like(normalized))
    count = valid.sum()
    # Mean squared VECTOR norm, matching offline normalized vector RMS.
    per_channel = masked.square().sum(-1).sum(tuple(range(valid.ndim))) / count.clamp_min(1)
    return dict(residual=residual, normalized=normalized, valid=valid, count=count,
                loss_by_channel=per_channel, loss=per_channel.sum(), has_valid=count > 0)


def discrete_constraints(*, r_B, v_B, S_B, F_average_B, tau_average_O_B,
                         x_O_W, u_O_W, omega_B, q_WB_wxyz, h, mass, gravity_W,
                         interior_dimensions, episode_id, state_valid, interval_valid,
                         history_valid, reference_W=None, interval_seconds=.005,
                         window_seconds=.2, window_intervals=40):
    """Return short and whole-window residuals/losses in channel order (r,P,H).

    mass [B,N], gravity [B,N,3], interior_dimensions [B,3], h [B,N-1].
    history_valid [B,N] must certify 249 causal 200 Hz held-input-valid samples
    ending at EACH endpoint (including the window start); the data layer builds
    this mask without crossing resets. It is never inferred from predictions.
    reference_W [B,3] must come from recorded robot state; default x_O_W[:,0].

    Raw residuals support varying positive durations. Default loss masks admit
    only full 5 ms short samples and 41-state/200 ms windows. Passing None for a
    duration/count restriction explicitly opts into irregular mathematical tests;
    partial intervals are otherwise excluded, not relabelled full samples.
    Invalid/empty batches return zero masked loss AND count=0/has_valid=False.
    Callers must not interpret the empty loss as validation success.
    """
    B, N, D = r_B.shape
    if D != 3 or N < 2:
        raise ValueError('Need [B,N>=2,3] endpoint heads')
    endpoints = (r_B, v_B, S_B, x_O_W, u_O_W, omega_B, gravity_W)
    if any(t.shape != (B,N,3) for t in endpoints) or q_WB_wxyz.shape != (B,N,4):
        raise ValueError('Endpoint tensor shapes differ')
    if any(t.shape != (B,N-1,3) for t in (F_average_B,tau_average_O_B)) or h.shape != (B,N-1):
        raise ValueError('Wrenches describe N-1 completed intervals')
    if any(t.shape != (B,N) for t in (mass,episode_id,state_valid,history_valid)) or interval_valid.shape != h.shape:
        raise ValueError('Mass, episode and validity shapes differ')
    if interior_dimensions.shape != (B,3):
        raise ValueError('Provide nominal interior dimensions [B,3]')
    if reference_W is None:
        reference_W = x_O_W[:,0]
    if reference_W.shape != (B,3):
        raise ValueError('Reference must be one fixed recorded world point per window')
    finite_state = torch.stack([torch.isfinite(t).all(-1) for t in endpoints],-1).all(-1)
    finite_state &= torch.isfinite(q_WB_wxyz).all(-1) & (q_WB_wxyz.square().sum(-1) > 1e-12)
    finite_state &= torch.isfinite(mass) & (mass > 0)
    finite_interval = torch.isfinite(h) & (h > 0)
    finite_interval &= torch.isfinite(F_average_B).all(-1) & torch.isfinite(tau_average_O_B).all(-1)
    # Invalid padded values cannot contaminate masked gradients through NaNs.
    clean = lambda t: torch.where(torch.isfinite(t), t, torch.zeros_like(t))
    r,v,S,x,u,w,g = (clean(t) for t in endpoints)
    m=clean(mass); dt=clean(h); q=clean(q_WB_wxyz)
    dims=clean(interior_dimensions); A=clean(reference_W)[:,None]
    R=quaternion_matrix(q)
    X=x+mv(R,r); U=u+mv(R,v+torch.linalg.cross(w,r)); P=m[...,None]*U
    H=mv(R,S)+torch.linalg.cross(X-A,P)
    IF=dt[...,None]*mv(R[:,1:],clean(F_average_B))
    Itau=dt[...,None]*mv(R[:,1:],clean(tau_average_O_B))+torch.linalg.cross(x[:,1:]-A,IF)
    mg=m[:,:1,None]*g[:,:1]
    G=dt[...,None]*torch.linalg.cross((X[:,:-1]+X[:,1:])/2-A,mg.expand_as(X[:,1:]))
    dr=dt[...,None]*(v[:,:-1]+v[:,1:])/2
    residual=torch.stack((r[:,1:]-r[:,:-1]-dr,
                          P[:,1:]-P[:,:-1]-dt[...,None]*mg+IF,
                          H[:,1:]-H[:,:-1]-G+Itau),-2)
    duration=dt.sum(-1)
    long_residual=torch.stack((r[:,-1]-r[:,0]-dr.sum(1),
        P[:,-1]-P[:,0]-duration[:,None]*mg[:,0]+IF.sum(1),
        H[:,-1]-H[:,0]-G.sum(1)+Itau.sum(1)),-2)
    L=torch.linalg.vector_norm(dims,dim=-1)
    gnorm=torch.linalg.vector_norm(g[:,0],dim=-1)
    F0=m[:,0]*gnorm; tau0=F0*L; u0=torch.sqrt((gnorm*L).clamp_min(1e-30))
    rates=torch.stack((u0,F0,tau0),-1)
    scales=(dt[...,None]*rates[:,None]).clamp_min(1e-30)
    long_scales=(duration[:,None]*rates).clamp_min(1e-30)
    constant_mass=torch.isclose(m,m[:,:1],rtol=1e-7,atol=1e-9).all(-1)
    uniform_gravity=torch.isclose(g,g[:,:1],rtol=1e-7,atol=1e-9).all(dim=(-2,-1))
    assumptions=constant_mass & uniform_gravity & (gnorm > 0)
    assumptions &= (torch.isfinite(interior_dimensions) & (interior_dimensions > 0)).all(-1)
    assumptions &= torch.isfinite(reference_W).all(-1) & finite_state.all(-1)
    endpoint_ok=state_valid.bool() & history_valid.bool() & finite_state
    valid=interval_valid.bool() & finite_interval & endpoint_ok[:,:-1] & endpoint_ok[:,1:]
    valid &= (episode_id[:,1:]==episode_id[:,:-1]) & assumptions[:,None]
    if interval_seconds is not None:
        valid &= torch.isclose(dt,dt.new_tensor(interval_seconds),rtol=0,atol=1e-8)
    window_valid=valid.all(-1)
    if window_seconds is not None:
        # Float32 summation of forty .005 values differs from .2 by ~1.5e-8.
        # Allow arithmetic roundoff, far below even one native partial step.
        tolerance=max(1e-8,4*torch.finfo(duration.dtype).eps*abs(window_seconds))
        window_valid &= torch.isclose(duration,duration.new_tensor(window_seconds),rtol=0,atol=tolerance)
    if window_intervals is not None:
        window_valid &= N-1 == window_intervals
    return dict(short=_result(residual,scales,valid),
                long=_result(long_residual,long_scales,window_valid),
                assumptions_valid=assumptions,constant_mass=constant_mass,uniform_gravity=uniform_gravity,
                X_W=X,U_W=U,P_W=P,H_A_W=H,force_impulse_W=IF,moment_impulse_A_W=Itau,
                gravity_impulse_A_W=G,reference_W=A[:,0],L=L,F0=F0,tau0=tau0,u0=u0)
