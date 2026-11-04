import copy
import numpy as np
import pytest
import torch
from liquid_tcn.ported.discrete.losses import discrete_constraints, quaternion_matrix, mv
from liquid_tcn.ported.discrete.data import environment_episodes,grid_intervals,eligible_windows,merged_rows

D=torch.float64

def fixture(n=41):
    zeros=torch.zeros(2,n,3,dtype=D)
    q=torch.zeros(2,n,4,dtype=D);q[...,0]=1
    g=zeros.clone();g[...,2]=-9.81
    return dict(r_B=zeros.clone(),v_B=zeros.clone(),S_B=zeros.clone(),
        F_average_B=g[:,1:].clone()*2,tau_average_O_B=zeros[:,1:].clone(),
        x_O_W=zeros.clone(),u_O_W=zeros.clone(),omega_B=zeros.clone(),q_WB_wxyz=q,
        h=torch.full((2,n-1),.005,dtype=D),mass=torch.full((2,n),2.,dtype=D),gravity_W=g,
        interior_dimensions=torch.tensor([[.3,.2,.4]]*2,dtype=D),
        episode_id=torch.zeros(2,n,dtype=torch.int64),state_valid=torch.ones(2,n,dtype=torch.bool),
        history_valid=torch.ones(2,n,dtype=torch.bool),interval_valid=torch.ones(2,n-1,dtype=torch.bool))

def static_offset(n=41):
    a=fixture(n);a['r_B'][...]=torch.tensor([.12,-.07,.2],dtype=D)
    a['tau_average_O_B']=torch.linalg.cross(a['r_B'][:,1:],a['F_average_B'])
    return a

def test_static_offset_gravity_moment_and_scales():
    a=static_offset();o=discrete_constraints(**a)
    torch.testing.assert_close(o['short']['residual'],torch.zeros_like(o['short']['residual']),atol=1e-14,rtol=0)
    assert o['long']['count']==2 and o['short']['count']==80
    assert o['L'][0]==pytest.approx(np.sqrt(.3**2+.2**2+.4**2))
    assert o['u0'][0]==pytest.approx(np.sqrt(9.81*float(o['L'][0])))
    a['tau_average_O_B'].zero_()
    assert discrete_constraints(**a)['long']['residual'][...,2,:].norm()>0

@pytest.mark.parametrize('reaction',[0.,7.])
@pytest.mark.parametrize('varying',[False,True])
def test_free_fall_and_known_applied_reaction(reaction,varying):
    a=fixture()
    if varying:a['h']*=torch.linspace(.5,1.5,40,dtype=D)
    t=torch.cat((torch.zeros(2,1,dtype=D),a['h'].cumsum(1)),1)
    force=torch.tensor([reaction,0.,0.],dtype=D)
    accel=a['gravity_W']-force/2
    a['r_B']=.5*t[...,None]**2*accel
    a['v_B']=t[...,None]*accel
    a['F_average_B']=force.expand(2,40,3).clone()
    a['tau_average_O_B']=torch.linalg.cross((a['r_B'][:,:-1]+a['r_B'][:,1:])/2,a['F_average_B'])
    o=discrete_constraints(**a,interval_seconds=None)
    torch.testing.assert_close(o['short']['residual'],torch.zeros_like(o['short']['residual']),atol=2e-15,rtol=0)
    assert o['long']['count']==2


def test_translating_rotating_frame_and_fixed_endpoint_origin():
    a=fixture();t=torch.arange(41,dtype=D)[None,:,None]*.005
    angle=2*t[...,0];a['q_WB_wxyz'][...,0]=torch.cos(angle/2);a['q_WB_wxyz'][...,3]=torch.sin(angle/2)
    R=quaternion_matrix(a['q_WB_wxyz']);Rt=R.transpose(-1,-2)
    a['u_O_W'][...]=torch.tensor([.7,-.3,.1],dtype=D)
    a['x_O_W']=t*a['u_O_W']+torch.tensor([10.,3.,1.],dtype=D)
    X=torch.tensor([10.2,3.1,1.4],dtype=D).expand_as(a['r_B'])
    a['r_B']=mv(Rt,X-a['x_O_W']);a['omega_B'][...,2]=2
    a['v_B']=mv(Rt,-a['u_O_W'])-torch.linalg.cross(a['omega_B'],a['r_B'])
    F=a['mass'][...,None]*a['gravity_W']
    a['F_average_B']=mv(Rt[:,1:],F[:,1:])
    a['tau_average_O_B']=mv(Rt[:,1:],torch.linalg.cross(X[:,1:]-a['x_O_W'][:,1:],F[:,1:]))
    o=discrete_constraints(**a)
    assert o['short']['residual'][...,1:,:].abs().max()<1e-14
    # A continuously-moving/midpoint origin is NOT the collector's convention.
    wrong=copy.deepcopy(a)
    wrong['tau_average_O_B']=mv(Rt[:,1:],torch.linalg.cross(X[:,1:]-(a['x_O_W'][:,1:]+a['x_O_W'][:,:-1])/2,F[:,1:]))
    assert discrete_constraints(**wrong)['long']['residual'][...,2,:].norm()>1e-4


def test_reference_covariance_and_zero_set_invariance():
    a=static_offset();A=torch.tensor([[2.,-.4,.1]]*2,dtype=D)
    o=discrete_constraints(**a);p=discrete_constraints(**a,reference_W=A)
    torch.testing.assert_close(o['long']['residual'],p['long']['residual'],atol=1e-14,rtol=0)
    a['F_average_B']+=.3
    o=discrete_constraints(**a);p=discrete_constraints(**a,reference_W=A)
    expected=o['long']['residual'][:,2]-torch.linalg.cross(A,o['long']['residual'][:,1])
    torch.testing.assert_close(p['long']['residual'][:,2],expected,atol=1e-14,rtol=0)


def test_short_sum_matches_long_and_partial_masks():
    a=static_offset();a['v_B']+=torch.randn(2,41,3,dtype=D)*.01
    a['h'][:,0]=.002
    o=discrete_constraints(**a)
    torch.testing.assert_close(o['short']['residual'].sum(1),o['long']['residual'],atol=1e-14,rtol=1e-12)
    assert o['short']['count']==78 and o['long']['count']==0
    assert not o['long']['has_valid'] and o['long']['loss']==0


def test_invalid_reset_history_mass_gravity_and_empty():
    for field in ('state_valid','history_valid'):
        a=fixture();a[field][:]=False
        o=discrete_constraints(**a);assert o['short']['count']==0 and not o['long']['has_valid']
    a=fixture();a['episode_id'][0,20:]=1;a['interval_valid'][1,4]=False
    o=discrete_constraints(**a);assert o['short']['count']==78 and o['long']['count']==0
    for field in ('mass','gravity_W'):
        a=fixture();a[field][0,10]+=1
        o=discrete_constraints(**a);assert not o['assumptions_valid'][0] and o['assumptions_valid'][1]
    a=fixture(40);assert discrete_constraints(**a)['long']['count']==0


def test_interval_average_can_differ_from_every_substep_force():
    a=fixture(2)
    native=torch.tensor([100.,-60.,80.,-20.,40.,-80.,20.,-40.,60.,0.],dtype=D)
    average=native.mean();impulse=native.sum()*.0005
    assert not bool((native==average).any())
    a['F_average_B'][...,0]=average
    a['v_B'][:,1,0]=-impulse/2
    a['r_B'][:,1,0]=.005*a['v_B'][:,1,0]/2
    a['tau_average_O_B']=torch.linalg.cross((a['r_B'][:,1:]+a['r_B'][:,:-1])/2,a['F_average_B'])
    o=discrete_constraints(**a)
    assert o['short']['residual'][...,1,:].abs().max()<1e-14


@pytest.mark.parametrize('horizon',['short','long'])
def test_double_gradcheck_and_all_head_gradients(horizon):
    torch.manual_seed(4);a=static_offset(3 if horizon=='short' else 41)
    heads=('r_B','v_B','S_B','F_average_B','tau_average_O_B')
    inputs=tuple((a[k]+torch.randn_like(a[k])*.01).requires_grad_() for k in heads)
    def f(*values):
        o=discrete_constraints(**dict(a,**dict(zip(heads,values))))
        return o[horizon]['normalized']
    assert torch.autograd.gradcheck(f,inputs,fast_mode=True,atol=1e-5,rtol=1e-4)
    f(*inputs).square().sum().backward()
    for value in inputs:assert torch.isfinite(value.grad).all() and value.grad.norm()>0
    a['r_B'][0,1,0]=float('nan');a['r_B'].requires_grad_()
    o=discrete_constraints(**a);o['short']['loss'].backward()
    assert torch.isfinite(a['r_B'].grad).all()


def row(tick,seq,ep=(0,0),n=0,reset=(False,False),terminal=(False,False)):
    a=static_offset(2);r={}
    for k in ('x_O_W','q_WB_wxyz','r_B','v_B','S_B','mass','gravity_W'):
        r[k]=a[k][:,0].numpy()
    r.update(velocity_W=np.zeros((2,6)),com_W=r['r_B'].copy(),com_vel_W=np.zeros((2,3)),
        momentum_W=np.zeros((2,3)),angular_momentum_world_origin_W=np.zeros((2,3)),
        state_valid=np.ones(2,bool),held_valid=np.ones(2,bool),episode_supported=np.ones(2,bool),episode_selected=np.ones(2,bool),
        episode_id=np.array(ep),episode_native_step=np.full(2,tick),native_step=tick,time_s=tick*.001,record_sequence=seq,
        reset=np.array(reset),terminal=np.array(terminal),interval_native_steps=np.full(2,n),interval_duration_s=np.full(2,n*.001),interval_valid=np.full(2,n>0))
    for k in ('coupling_force_impulse_W','coupling_moment_impulse_world_origin_W','force_interval_average_B','moment_interval_average_O_B','momentum_balance_moment_impulse_world_origin_W'):
        r[k]=np.zeros((2,3))
    return r


def test_partial_reset_repeated_timestamp_terminal_and_incomplete_history():
    rows=[row(0,1,reset=(True,True)),row(2,2,n=2),row(2,3,terminal=(False,True)),
          row(2,4,ep=(0,1),reset=(False,True)),row(5,5,ep=(0,1),n=3),row(5,6,ep=(0,1),terminal=(True,True))]
    e=environment_episodes(rows,0,.001)[0];grid,rejected=grid_intervals(e,.001)
    assert len(grid)==1 and grid[0]['valid'] and len(grid[0]['parts'])==2
    assert rejected['merged_split_intervals']==1 and e['terminals']==[5]
    assert eligible_windows(grid,e['nodes'],5)==([],[],0)
    episodes=environment_episodes(rows,1,.001)
    assert episodes[0]['terminals']==[2] and 2 in episodes[0]['nodes']
    assert not grid_intervals(episodes[0],.001)[0] and not grid_intervals(episodes[1],.001)[0]
    rows[4]['interval_duration_s'][0]=.004
    with pytest.raises(ValueError,match='duration'):environment_episodes(rows,0,.001)


def test_merge_stream_copies_and_conflicts(monkeypatch):
    import liquid_tcn.ported.discrete.data as d
    full=row(0,1);event={k:full[k] for k in ('native_step','record_sequence','time_s','episode_id')}
    monkeypatch.setattr(d,'stream_rows',lambda p: iter([event] if p.name=='events' else [] if p.name=='boundaries' else [full]))
    assert len(merged_rows('fixture'))==1
    event['episode_id']=np.array([0,1])
    with pytest.raises(ValueError,match='Conflicting'):merged_rows('fixture')


def test_native_gravity_independent_of_precomputed_balance_fields():
    from liquid_tcn.ported.discrete.analyze import prepare_intervals,evaluate_windows,CRITERIA
    dt=.001;g=np.array([0.,0.,-9.81]);u=np.array([.8,.3,0.]);x=np.array([.2,-.1,.3]);rows=[]
    for i in range(11):
        if i:u=u+g*dt;x=x+u*dt  # inspected kick-then-drift convention
        s=row(i,i+1,n=int(i>0))
        s['r_B'][:]=x;s['v_B'][:]=u;s['com_W'][:]=x;s['com_vel_W'][:]=u
        s['momentum_W'][:]=2*u;s['angular_momentum_world_origin_W'][:]=np.cross(x,2*u)
        # Corrupt this diagnostic on purpose: it MUST NOT drive the native floor.
        s['momentum_balance_moment_impulse_world_origin_W'][:]=[1.,2.,3.]
        rows.append(s)
    cr=[copy.deepcopy(rows[0])]
    for i in (5,10):
        s=copy.deepcopy(rows[i]);s['interval_native_steps'][:]=5;s['interval_duration_s'][:]=.005;cr.append(s)
    ne=environment_episodes(rows,0,dt)[0];ce=environment_episodes(cr,0,dt)[0]
    grid,_,anchor=prepare_intervals(ce,ne,dt,CRITERIA)
    result=evaluate_windows([[p] for p in grid],ce,ne,anchor,np.array([.3,.2,.4]),'short')
    angular=result['channels']['angular_momentum']
    assert angular['native_floor']['rms']<1e-15
    assert angular['gravity_quadrature']['rms']>1e-6
    assert angular['aggregation_frame']['rms']<1e-15
    assert result['gravity_recovery_crosscheck']['rms']>1
    expected=-.5*dt*.005*np.cross(np.array([.8,.3,0.]),2*g)
    np.testing.assert_allclose(angular['total']['bias'],expected,atol=1e-15)


def test_split_interval_impulses_merge_in_common_world_reference():
    from liquid_tcn.ported.discrete.analyze import prepare_intervals,CRITERIA
    rows=[row(0,1),row(2,2,n=2),row(2,3,terminal=(False,True)),
          row(2,4,ep=(0,1),reset=(False,True)),row(5,5,ep=(0,1),n=3)]
    forces=[np.array([.1,.2,-.3]),np.array([-.4,.5,.2])]
    moments=[np.array([.01,-.03,.04]),np.array([-.05,.06,.02])]
    for i,f,j,theta in zip((1,4),forces,moments,(.3,.8)):
        s=rows[i];q=np.array([np.cos(theta/2),0.,0.,np.sin(theta/2)])
        s['q_WB_wxyz'][0]=q;s['x_O_W'][0]=[i*.1,-i*.2,0.]
        R=quaternion_matrix(torch.tensor(q)).numpy();h=float(s['interval_duration_s'][0])
        s['force_interval_average_B'][0]=R.T@f/h
        s['moment_interval_average_O_B'][0]=R.T@(j-np.cross(s['x_O_W'][0],f))/h
        s['coupling_force_impulse_W'][0]=f;s['coupling_moment_impulse_world_origin_W'][0]=j
    for i in (2,3):
        for k in ('q_WB_wxyz','x_O_W'):rows[i][k][0]=rows[1][k][0]
    ep=environment_episodes(rows,0,.001)[0];grid,_,_=prepare_intervals(ep,ep,.001,CRITERIA)
    assert len(grid)==1 and grid[0]['valid']
    end=ep['nodes'][5];R=quaternion_matrix(torch.tensor(end['q_WB_wxyz'])).numpy()
    f=.005*(R@grid[0]['F_B']);j=.005*(R@grid[0]['tau_B'])+np.cross(end['x_O_W'],f)
    np.testing.assert_allclose(f,sum(forces),atol=1e-15)
    np.testing.assert_allclose(j,sum(moments),atol=1e-15)


def test_no_success_from_empty_statistics_or_incomplete_grid():
    from liquid_tcn.ported.discrete.analyze import statistics,verdict,CRITERIA
    empty=statistics([],[])
    assert verdict(empty,empty,True,CRITERIA)['status']=='inconclusive'
    rows=[row(0,1),row(5,2,n=5),row(15,3,n=5)]
    ep=environment_episodes(rows,0,.001)[0];grid,_=grid_intervals(ep,.001)
    short,long,histories=eligible_windows(grid,ep['nodes'],5,history_taps=2,long_intervals=2)
    assert not short and not long and histories==1


def test_float32_full_window_is_not_rejected_by_duration_roundoff():
    a={k:v.float() if v.dtype==torch.float64 else v for k,v in static_offset().items()}
    o=discrete_constraints(**a)
    assert o['short']['count']==80 and o['long']['count']==2
    assert torch.isfinite(o['long']['loss'])
    a['h'][:,0]=.0045
    assert discrete_constraints(**a)['long']['count']==0
