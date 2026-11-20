from osmctx.boundary import bbox, square_boundary, to_polygon
from osmctx.fetch import _tile_url, boundary_hash, build_query, count_categories


def test_query_has_categories():
    q = build_query(bbox(to_polygon(square_boundary(48.5193, 9.0565, 100))))
    for k in ('"building"', '"highway"', '"tree"', '"tree_row"', '"wood"', '"forest"',
              '"waterway"', '"water"', '"coastline"', '"beach"', '"wetland"', "out geom;"):
        assert k in q


def test_tile_names():
    assert _tile_url(48, 9).endswith("Copernicus_DSM_COG_10_N48_00_E009_00_DEM.tif")
    assert "S09_00_W078" in _tile_url(-9, -78)
    assert "S09_00_E115" in _tile_url(-9, 115)


def test_hash_deterministic():
    assert boundary_hash(square_boundary(48.5193, 9.0565, 100)) == \
        boundary_hash(square_boundary(48.5193, 9.0565, 100))


def test_count():
    osm = {"elements": [{"tags": {"building": "yes"}}, {"tags": {"natural": "tree"}},
                        {"tags": {"waterway": "river"}}]}
    c = count_categories(osm)
    assert c["building"] == 1 and c["tree"] == 1 and c["waterway"] == 1
