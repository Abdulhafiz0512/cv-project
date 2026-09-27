"""Rule tests on synthetic trajectories placed in the real scene layout."""
from helpers import context, events, make_track, norm


def labels(evts):
    return [e[2] for e in evts]


def test_jaywalking_mid_block_but_not_on_zebra():
    walker = make_track(1, 0, 0.0, [(0.0, norm(0.10, 0.30)), (8.0, norm(0.40, 0.24))])   # across outbound lanes
    zebra = make_track(2, 0, 0.0, [(0.0, norm(0.20, 0.61)), (8.0, norm(0.58, 0.52))])    # along the west zebra
    ev = events.jaywalking(context([walker, zebra], 10.0))
    assert len(ev) == 1
    s, e = ev[0]
    assert 0.0 <= s < 3.0 and e > 5.0
    assert events.jaywalking(context([zebra], 10.0)) == []


def test_wrong_way_on_inbound_carriageway():
    # inbound traffic flows to the bottom-right; this car drives to the top-left
    car = make_track(1, 2, 0.0, [(0.0, norm(0.40, 0.40)), (4.0, norm(0.12, 0.23))])
    ev = events.wrong_way(context([car], 6.0))
    assert len(ev) == 1 and ev[0][1] - ev[0][0] > 2.0
    legal = make_track(2, 2, 0.0, [(0.0, norm(0.12, 0.23)), (4.0, norm(0.40, 0.40))])
    assert events.wrong_way(context([legal], 6.0)) == []


def test_stopped_vehicle_needs_ten_seconds():
    p = norm(0.20, 0.12)  # outbound curb lane, far from the junction and the bus bay
    parked = make_track(1, 2, 0.0, [(0.0, p), (15.0, p)])
    ev = events.stopped_vehicle(context([parked], 20.0))
    assert len(ev) == 1 and ev[0][1] - ev[0][0] >= 10.0
    short = make_track(2, 2, 0.0, [(0.0, p), (7.0, p)])
    assert events.stopped_vehicle(context([short], 20.0)) == []


def test_queue_is_not_a_stopped_vehicle():
    a = make_track(1, 2, 0.0, [(0.0, norm(0.20, 0.12)), (15.0, norm(0.20, 0.12))])
    b = make_track(2, 2, 0.0, [(0.0, norm(0.23, 0.135)), (15.0, norm(0.23, 0.135))])
    assert events.stopped_vehicle(context([a, b], 20.0)) == []


def test_red_light_runner_passes_a_waiting_car():
    waiting = make_track(1, 2, 0.0, [(0.0, norm(0.40, 0.44)), (20.0, norm(0.40, 0.44))])   # at the line
    runner = make_track(2, 2, 5.0, [(5.0, norm(0.10, 0.36)), (9.0, norm(0.45, 0.62)), (11.0, norm(0.65, 0.70))])
    ev = events.red_light(context([waiting, runner], 20.0))
    assert len(ev) == 1
    assert 6.0 < ev[0][0] < 9.0
    # the same crossing with nobody waiting is a green phase
    assert events.red_light(context([runner], 20.0)) == []


def test_u_turn_in_junction():
    import numpy as np
    c = norm(0.60, 0.65)
    ts = np.linspace(0, 5, 11)
    pts = [(float(t), c + 120 * np.array([np.cos(np.pi * t / 5), -np.sin(np.pi * t / 5)])) for t in ts]
    lead_in = [(-2.0, c + np.array([120.0, 250.0]))]
    lead_out = [(7.0, c + np.array([-120.0, 250.0]))]
    car = make_track(1, 2, -2.0, lead_in + pts + lead_out)
    ev = events.illegal_u_turn(context([car], 10.0))
    assert len(ev) == 1


def test_failure_to_yield_on_west_zebra():
    ped = make_track(1, 0, 0.0, [(0.0, norm(0.30, 0.575)), (6.0, norm(0.40, 0.555))])
    car = make_track(2, 2, 1.0, [(1.0, norm(0.28, 0.45)), (3.0, norm(0.34, 0.60)), (4.0, norm(0.37, 0.70))])
    assert labels([[s, e, "failure_to_yield"] for s, e in events.failure_to_yield(context([ped, car], 8.0))]) \
        == ["failure_to_yield"]
    far_ped = make_track(3, 0, 0.0, [(0.0, norm(0.56, 0.51)), (6.0, norm(0.59, 0.505))])
    assert events.failure_to_yield(context([far_ped, car], 8.0)) == []


def test_run_rules_output_format():
    car = make_track(1, 2, 0.0, [(0.0, norm(0.40, 0.40)), (4.0, norm(0.12, 0.23))])
    out = events.run_rules(context([car], 6.0))
    for s, e, lab in out:
        assert 0 <= s < e <= 6.0 and isinstance(lab, str)


def test_accident_needs_contact_and_abrupt_stop():
    import numpy as np
    p = norm(0.75, 0.62)  # junction, outside the signal queue
    struck = make_track(1, 2, 0.0, [(0.0, p), (12.0, p)])
    striker = make_track(2, 2, 0.0, [(0.0, p - np.array([480.0, 0])), (2.0, p - np.array([40.0, 0])),
                                      (12.0, p - np.array([40.0, 0]))])
    ev = events.accident(context([struck, striker], 12.0))
    assert len(ev) == 1 and 1.0 < ev[0][0] < 2.5
    # stopping a car length short of the other vehicle is not a collision
    careful = make_track(3, 2, 0.0, [(0.0, p - np.array([480.0, 0])), (2.0, p - np.array([130.0, 0])),
                                     (12.0, p - np.array([130.0, 0]))])
    assert events.accident(context([struck, careful], 12.0)) == []


def test_fast_person_is_a_scooter_rider_not_a_jaywalker():
    # 6 bh/s along the inbound lanes: far faster than anyone on foot
    rider = make_track(1, 0, 0.0, [(0.0, norm(0.05, 0.20)), (3.0, norm(0.45, 0.42))])
    assert events.jaywalking(context([rider], 5.0)) == []


def test_pushed_bike_on_zebra_is_not_failure_to_yield():
    ped = make_track(1, 0, 0.0, [(0.0, norm(0.30, 0.575)), (6.0, norm(0.40, 0.555))])
    pushed = make_track(2, 3, 0.0, [(0.0, norm(0.28, 0.58)), (6.0, norm(0.38, 0.56))])  # walking pace
    assert events.failure_to_yield(context([ped, pushed], 8.0)) == []
