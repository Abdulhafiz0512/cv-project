from helpers import scene, norm  # noqa: F401  (sets sys.path)

import numpy as np

from trafficev import segments


def test_union_and_gaps():
    assert segments.union([(0, 2), (1, 3), (5, 6)]) == [(0, 3), (5, 6)]
    assert segments.close_gaps([(0, 2), (2.4, 3)], 0.5) == [(0, 3)]
    assert segments.close_gaps([(0, 2), (3, 4)], 0.5) == [(0, 2), (3, 4)]
    assert segments.drop_short([(0, 0.2), (1, 3)], 0.5) == [(1, 3)]
    assert segments.clip([(-1, 2), (9, 12)], 10) == [(0, 2), (9, 10)]


def test_flags_to_intervals():
    t = np.arange(0, 1.0, 0.1)
    f = np.array([0, 1, 1, 0, 0, 1, 1, 1, 1, 1], bool)
    out = segments.flags_to_intervals(t, f, 0.1)
    assert np.allclose(out, [(0.1, 0.3), (0.5, 1.0)])


def test_tiou():
    assert segments.tiou((0, 10), (5, 15)) == 5 / 15
    assert segments.tiou((0, 1), (2, 3)) == 0.0


def test_scene_regions():
    # zebra centres are on a crosswalk and on the road
    for poly in scene.CROSSWALKS.values():
        c = poly.mean(axis=0)[None]
        assert scene.on_crosswalk(c)[0]
        assert scene.on_road(c)[0]
    # the median and the islands are not carriageway
    assert not scene.on_road(scene.MEDIAN.mean(axis=0)[None])[0]
    # the bus-stop sidewalk (top) is off-road; the inbound lanes are on it
    assert not scene.on_road(norm(0.55, 0.17)[None])[0]
    assert scene.on_road(norm(0.30, 0.40)[None])[0]
    assert scene.one_way_zone(norm(0.30, 0.40)[None])[0] == 0      # inbound
    assert scene.one_way_zone(norm(0.30, 0.20)[None])[0] == 1      # outbound


def test_stop_line_orientation():
    from trafficev import signals
    upstream = norm(0.25, 0.42)[None]      # before the line, in the inbound lanes
    downstream = norm(0.30, 0.56)[None]    # on the zebra, past the line
    assert signals.distance_upstream(upstream)[0] > 0
    assert signals.distance_upstream(downstream)[0] < 0


def test_road_depth_is_zero_off_road():
    assert scene.road_depth(norm(0.55, 0.17)[None])[0] == 0.0
    assert scene.road_depth(norm(0.30, 0.40)[None])[0] > 20
