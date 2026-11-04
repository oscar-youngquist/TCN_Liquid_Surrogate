"""Conservative interval reduction in a fixed world frame, before host transfer."""
import torch
from .physics import rotate


IMPULSES = (
    'coupling_force_impulse_W', 'coupling_moment_impulse_world_origin_W',
    'momentum_balance_force_impulse_W', 'momentum_balance_moment_impulse_world_origin_W',
    'force_closure_residual_impulse_W', 'moment_closure_residual_impulse_W',
)
ALL_FLAGS = ('finite', 'contained', 'all_active', 'attached', 'inside_nominal_cavity')


def interval_fields(row, dt, count):
    """Uniform duration/diagnostic schema, including zero-duration event records."""
    row = dict(row)
    row['interval_duration_s'] = torch.full_like(row['mass'], dt)
    row['interval_native_steps'] = torch.full_like(row['episode_id'], count)
    for key in ALL_FLAGS:
        row['interval_'+key] = row[key]
    row['interval_domain_contact'] = row['domain_contact']
    row['interval_max_wall_penetration_m'] = row['max_wall_penetration_m']
    row['interval_min_active_count'] = row['active_count']
    return row


class GridAccumulator:
    def __init__(self, native_dt):
        self.dt = native_dt
        self.count = 0

    def push(self, row, start_velocity, tick, sequence):
        if self.count == 0:
            self.sums = {k: row[k].double().clone() for k in IMPULSES}
            self.valid = row['interval_valid'].clone()
            self.all_flags = {k: row[k].clone() for k in ALL_FLAGS}
            self.domain_contact = row['domain_contact'].clone()
            self.penetration = row['max_wall_penetration_m'].clone()
            self.active_count = row['active_count'].clone()
            self.start_velocity = start_velocity.clone()
        else:
            for k in IMPULSES:
                self.sums[k] += row[k]
            self.valid &= row['interval_valid']
            for k in ALL_FLAGS:
                self.all_flags[k] &= row[k]
            self.domain_contact |= row['domain_contact']
            self.penetration = torch.maximum(self.penetration, row['max_wall_penetration_m'])
            self.active_count = torch.minimum(self.active_count, row['active_count'])
        self.count += 1
        self.last, self.tick, self.sequence = row, tick, sequence

    def pop(self):
        if not self.count:
            return None
        duration = self.count*self.dt
        row = interval_fields(self.last, duration, self.count)
        row.update({k: v.to(self.last[k].dtype) for k, v in self.sums.items()})
        row['interval_valid'] = self.valid
        for k, value in self.all_flags.items():
            row['interval_'+k] = value
        row['interval_domain_contact'] = self.domain_contact
        row['interval_max_wall_penetration_m'] = self.penetration
        row['interval_min_active_count'] = self.active_count
        # Shift the summed fixed-world-origin moment to ONE fixed reference:
        # the endpoint base origin. Never sum vectors in changing base frames.
        origin, q = row['x_O_W'], row['q_WB_wxyz']
        force = row['coupling_force_impulse_W']
        moment = row['coupling_moment_impulse_world_origin_W']-torch.linalg.cross(origin, force)
        row.update(coupling_origin_W=origin, coupling_q_WB_wxyz=q,
                   coupling_moment_impulse_O_W=moment,
                   coupling_force_impulse_B=rotate(q, force, True),
                   coupling_moment_impulse_O_B=rotate(q, moment, True),
                   base_acceleration_interval_average_W=(row['velocity_W']-self.start_velocity)/duration)
        row['force_interval_average_B'] = row['coupling_force_impulse_B']/duration
        row['moment_interval_average_O_B'] = row['coupling_moment_impulse_O_B']/duration
        self.count = 0
        return row, self.tick, self.sequence
